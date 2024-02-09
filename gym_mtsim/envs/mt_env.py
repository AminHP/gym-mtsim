from typing import List, Tuple, Dict, Any, Optional, Union, Callable

import copy
from datetime import datetime
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
from scipy.special import expit

import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_colors
import plotly.graph_objects as go

import gymnasium as gym
from gymnasium import spaces

from ..simulator import MtSimulator, OrderType


class MtEnv(gym.Env):

    metadata = {'render_modes': ['human', 'simple_figure', 'advanced_figure']}

    def __init__(
        self,
        original_simulator: MtSimulator,
        trading_symbols: List[str],
        window_size: int,
        time_points: Optional[List[datetime]] = None,
        hold_threshold: float = 0.5,
        close_threshold: float = 0.5,
        fee: Union[float, Callable[[str], float]] = 0.0005,
        symbol_max_orders: int = 1,
        multiprocessing_processes: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        # validations
        assert len(original_simulator.symbols_data) > 0, "no data available"
        assert len(original_simulator.symbols_info) > 0, "no data available"
        assert len(trading_symbols) > 0, "no trading symbols provided"
        assert 0. <= hold_threshold <= 1., "'hold_threshold' must be in range [0., 1.]"

        if not original_simulator.hedge:
            symbol_max_orders = 1

        for symbol in trading_symbols:
            assert symbol in original_simulator.symbols_info, f"symbol '{symbol}' not found"
            currency_profit = original_simulator.symbols_info[symbol].currency_profit
            assert original_simulator._get_unit_symbol_info(currency_profit) is not None, \
                   f"unit symbol for '{currency_profit}' not found"

        if time_points is None:
            time_points = original_simulator.symbols_data[trading_symbols[0]].index.to_pydatetime().tolist()
        assert len(time_points) > window_size, "not enough time points provided"

        self.render_mode = render_mode

        # attributes
        self.original_simulator = original_simulator
        self.trading_symbols = trading_symbols
        self.window_size = window_size
        self.time_points = time_points
        self.hold_threshold = hold_threshold
        self.close_threshold = close_threshold
        self.fee = fee
        self.symbol_max_orders = symbol_max_orders
        self.multiprocessing_pool = Pool(multiprocessing_processes) if multiprocessing_processes else None

        self.prices = self._get_prices()
        self.signal_features = self._process_data()
        self.features_shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Box(
            low=-1e2, high=1e2, dtype=np.float64,
            shape=(len(self.trading_symbols) * (self.symbol_max_orders + 2),)
        )  # symbol -> [close_order_i(logit), hold(logit), volume]

        INF = 1e10
        self.observation_space = spaces.Dict({
            'balance': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'equity': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'margin': spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
            'features': spaces.Box(low=-INF, high=INF, shape=self.features_shape, dtype=np.float64),
            'orders': spaces.Box(
                low=-INF, high=INF, dtype=np.float64,
                shape=(len(self.trading_symbols), self.symbol_max_orders, 3)
            )  # symbol, order_i -> [entry_price, volume, profit]
        })

        # episode
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.time_points) - 1
        self._truncated: bool = NotImplemented
        self._current_tick: int = NotImplemented
        self.simulator: MtSimulator = NotImplemented
        self.history: List[Dict[str, Any]] = NotImplemented

    def reset(self, seed=None, options=None) -> Dict[str, np.ndarray]:
        super().reset(seed=seed, options=options)

        self._truncated = False
        self._current_tick = self._start_tick
        self.simulator = copy.deepcopy(self.original_simulator)
        self.simulator.current_time = self.time_points[self._current_tick]
        self.history = [self._create_info()]

        observation = self._get_observation()
        info = self._create_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        orders_info, closed_orders_info = self._apply_action(action)

        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._truncated = True

        dt = self.time_points[self._current_tick] - self.time_points[self._current_tick - 1]
        self.simulator.tick(dt)

        step_reward = self._calculate_reward()

        info = self._create_info(
            orders=orders_info, closed_orders=closed_orders_info, step_reward=step_reward
        )
        observation = self._get_observation()
        self.history.append(info)

        return observation, step_reward, False, self._truncated, info

    def _apply_action(self, action: np.ndarray) -> Tuple[Dict, Dict]:
        orders_info = {}
        closed_orders_info = {symbol: [] for symbol in self.trading_symbols}

        k = self.symbol_max_orders + 2

        for i, symbol in enumerate(self.trading_symbols):
            symbol_action = action[k*i:k*(i+1)]
            close_orders_logit = symbol_action[:-2]
            hold_logit = symbol_action[-2]
            volume = symbol_action[-1]

            close_orders_probability = expit(close_orders_logit)
            hold_probability = expit(hold_logit)
            hold = bool(hold_probability > self.hold_threshold)
            modified_volume = self._get_modified_volume(symbol, volume)

            symbol_orders = self.simulator.symbol_orders(symbol)
            orders_to_close_index = np.where(
                close_orders_probability[:len(symbol_orders)] > self.close_threshold
            )[0]
            orders_to_close = np.array(symbol_orders)[orders_to_close_index]

            for j, order in enumerate(orders_to_close):
                self.simulator.close_order(order)
                closed_orders_info[symbol].append(dict(
                    order_id=order.id, symbol=order.symbol, order_type=order.type,
                    volume=order.volume, fee=order.fee,
                    margin=order.margin, profit=order.profit,
                    close_probability=close_orders_probability[orders_to_close_index][j],
                ))

            orders_capacity = self.symbol_max_orders - (len(symbol_orders) - len(orders_to_close))
            orders_info[symbol] = dict(
                order_id=None, symbol=symbol, hold_probability=hold_probability,
                hold=hold, volume=volume, capacity=orders_capacity, order_type=None,
                modified_volume=modified_volume, fee=float('nan'), margin=float('nan'),
                error='',
            )

            if self.simulator.hedge and orders_capacity == 0:
                orders_info[symbol].update(dict(
                    error="cannot add more orders"
                ))
            elif not hold:
                order_type = OrderType.Buy if volume > 0. else OrderType.Sell
                fee = self.fee if type(self.fee) is float else self.fee(symbol)

                try:
                    order = self.simulator.create_order(order_type, symbol, modified_volume, fee)
                    new_info = dict(
                        order_id=order.id, order_type=order_type,
                        fee=fee, margin=order.margin,
                    )
                except ValueError as e:
                    new_info = dict(error=str(e))

                orders_info[symbol].update(new_info)

        return orders_info, closed_orders_info

    def _get_prices(self, keys: List[str]=['Close', 'Open']) -> Dict[str, np.ndarray]:
        prices = {}

        for symbol in self.trading_symbols:
            get_price_at = lambda time: \
                self.original_simulator.price_at(symbol, time)[keys]

            if self.multiprocessing_pool is None:
                p = list(map(get_price_at, self.time_points))
            else:
                p = self.multiprocessing_pool.map(get_price_at, self.time_points)

            prices[symbol] = np.array(p)

        return prices

    def _process_data(self) -> np.ndarray:
        data = self.prices
        signal_features = np.column_stack(list(data.values()))
        return signal_features

    def _get_observation(self) -> Dict[str, np.ndarray]:
        features = self.signal_features[(self._current_tick-self.window_size+1):(self._current_tick+1)]

        orders = np.zeros(self.observation_space['orders'].shape)
        for i, symbol in enumerate(self.trading_symbols):
            symbol_orders = self.simulator.symbol_orders(symbol)
            for j, order in enumerate(symbol_orders):
                orders[i, j] = [order.entry_price, order.volume, order.profit]

        observation = {
            'balance': np.array([self.simulator.balance]),
            'equity': np.array([self.simulator.equity]),
            'margin': np.array([self.simulator.margin]),
            'features': features,
            'orders': orders,
        }
        return observation

    def _calculate_reward(self) -> float:
        prev_equity = self.history[-1]['equity']
        current_equity = self.simulator.equity
        step_reward = current_equity - prev_equity
        return step_reward

    def _create_info(self, **kwargs: Any) -> Dict[str, Any]:
        info = {k: v for k, v in kwargs.items()}
        info['balance'] = self.simulator.balance
        info['equity'] = self.simulator.equity
        info['margin'] = self.simulator.margin
        info['free_margin'] = self.simulator.free_margin
        info['margin_level'] = self.simulator.margin_level
        return info

    def _get_modified_volume(self, symbol: str, volume: float) -> float:
        si = self.simulator.symbols_info[symbol]
        v = abs(volume)
        v = np.clip(v, si.volume_min, si.volume_max)
        v = round(v / si.volume_step) * si.volume_step
        return v

    def render(self, mode: str='human', **kwargs: Any) -> Any:
        if mode == 'simple_figure':
            return self._render_simple_figure(**kwargs)
        if mode == 'advanced_figure':
            return self._render_advanced_figure(**kwargs)
        return self.simulator.get_state(**kwargs)

    def _render_simple_figure(
        self, figsize: Tuple[float, float]=(14, 6), return_figure: bool=False
    ) -> Any:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.trading_symbols)))

        for j, symbol in enumerate(self.trading_symbols):
            close_price = self.prices[symbol][:, 0]
            symbol_color = symbol_colors[j]

            ax.plot(self.time_points, close_price, c=symbol_color, marker='.', label=symbol)

            buy_ticks = []
            buy_error_ticks = []
            sell_ticks = []
            sell_error_ticks = []
            close_ticks = []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1

                order = self.history[i]['orders'].get(symbol, {})
                if order and not order['hold']:
                    if order['order_type'] == OrderType.Buy:
                        if order['error']:
                            buy_error_ticks.append(tick)
                        else:
                            buy_ticks.append(tick)
                    else:
                        if order['error']:
                            sell_error_ticks.append(tick)
                        else:
                            sell_ticks.append(tick)

                closed_orders = self.history[i]['closed_orders'].get(symbol, [])
                if len(closed_orders) > 0:
                    close_ticks.append(tick)

            tp = np.array(self.time_points)
            ax.plot(tp[buy_ticks], close_price[buy_ticks], '^', color='green')
            ax.plot(tp[buy_error_ticks], close_price[buy_error_ticks], '^', color='gray')
            ax.plot(tp[sell_ticks], close_price[sell_ticks], 'v', color='red')
            ax.plot(tp[sell_error_ticks], close_price[sell_error_ticks], 'v', color='gray')
            ax.plot(tp[close_ticks], close_price[close_ticks], '|', color='black')

            ax.tick_params(axis='y', labelcolor=symbol_color)
            ax.yaxis.tick_left()
            if j < len(self.trading_symbols) - 1:
                ax = ax.twinx()

        fig.suptitle(
            f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} ~ "
            f"Equity: {self.simulator.equity:.6f} ~ "
            f"Margin: {self.simulator.margin:.6f} ~ "
            f"Free Margin: {self.simulator.free_margin:.6f} ~ "
            f"Margin Level: {self.simulator.margin_level:.6f}"
        )
        fig.legend(loc='right')

        if return_figure:
            return fig

        plt.show()

    def _render_advanced_figure(
        self,
        figsize: Tuple[float, float] = (1400, 600),
        time_format: str = "%Y-%m-%d %H:%m",
        return_figure: bool = False,
    ) -> Any:
        fig = go.Figure()

        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.trading_symbols)))
        get_color_string = lambda color: "rgba(%s, %s, %s, %s)" % tuple(color)

        extra_info = [
            f"balance: {h['balance']:.6f} {self.simulator.unit}<br>"
            f"equity: {h['equity']:.6f}<br>"
            f"margin: {h['margin']:.6f}<br>"
            f"free margin: {h['free_margin']:.6f}<br>"
            f"margin level: {h['margin_level']:.6f}"
            for h in self.history
        ]
        extra_info = [extra_info[0]] * (self.window_size - 1) + extra_info

        for j, symbol in enumerate(self.trading_symbols):
            close_price = self.prices[symbol][:, 0]
            symbol_color = symbol_colors[j]

            fig.add_trace(
                go.Scatter(
                    x=self.time_points,
                    y=close_price,
                    mode='lines+markers',
                    line_color=get_color_string(symbol_color),
                    opacity=1.0,
                    hovertext=extra_info,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    legendgroup=f'g{j+1}',
                ),
            )

            fig.update_layout(**{
                f'yaxis{j+1}': dict(
                    tickfont=dict(color=get_color_string(symbol_color * [1, 1, 1, 0.8])),
                    overlaying='y' if j > 0 else None,
                    # position=0.035*j
                ),
            })

            trade_ticks = []
            trade_markers = []
            trade_colors = []
            trade_sizes = []
            trade_extra_info = []
            trade_max_volume = max([
                h.get('orders', {}).get(symbol, {}).get('modified_volume') or 0
                for h in self.history
            ])
            close_ticks = []
            close_extra_info = []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1

                order = self.history[i]['orders'].get(symbol)
                if order and not order['hold']:
                    marker = None
                    color = None
                    size = 8 + 22 * (order['modified_volume'] / trade_max_volume)
                    info = (
                        f"order id: {order['order_id'] or ''}<br>"
                        f"hold probability: {order['hold_probability']:.4f}<br>"
                        f"hold: {order['hold']}<br>"
                        f"volume: {order['volume']:.6f}<br>"
                        f"modified volume: {order['modified_volume']:.4f}<br>"
                        f"fee: {order['fee']:.6f}<br>"
                        f"margin: {order['margin']:.6f}<br>"
                        f"error: {order['error']}"
                    )

                    if order['order_type'] == OrderType.Buy:
                        marker = 'triangle-up'
                        color = 'gray' if order['error'] else 'green'
                    else:
                        marker = 'triangle-down'
                        color = 'gray' if order['error'] else 'red'

                    trade_ticks.append(tick)
                    trade_markers.append(marker)
                    trade_colors.append(color)
                    trade_sizes.append(size)
                    trade_extra_info.append(info)

                closed_orders = self.history[i]['closed_orders'].get(symbol, [])
                if len(closed_orders) > 0:
                    info = []
                    for order in closed_orders:
                        info_i = (
                            f"order id: {order['order_id']}<br>"
                            f"order type: {order['order_type'].name}<br>"
                            f"close probability: {order['close_probability']:.4f}<br>"
                            f"margin: {order['margin']:.6f}<br>"
                            f"profit: {order['profit']:.6f}"
                        )
                        info.append(info_i)
                    info = '<br>---------------------------------<br>'.join(info)

                    close_ticks.append(tick)
                    close_extra_info.append(info)

            fig.add_trace(
                go.Scatter(
                    x=np.array(self.time_points)[trade_ticks],
                    y=close_price[trade_ticks],
                    mode='markers',
                    hovertext=trade_extra_info,
                    marker_symbol=trade_markers,
                    marker_color=trade_colors,
                    marker_size=trade_sizes,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    showlegend=False,
                    legendgroup=f'g{j+1}',
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=np.array(self.time_points)[close_ticks],
                    y=close_price[close_ticks],
                    mode='markers',
                    hovertext=close_extra_info,
                    marker_symbol='line-ns',
                    marker_color='black',
                    marker_size=7,
                    marker_line_width=1.5,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    showlegend=False,
                    legendgroup=f'g{j+1}',
                ),
            )

        title = (
            f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} ~ "
            f"Equity: {self.simulator.equity:.6f} ~ "
            f"Margin: {self.simulator.margin:.6f} ~ "
            f"Free Margin: {self.simulator.free_margin:.6f} ~ "
            f"Margin Level: {self.simulator.margin_level:.6f}"
        )
        fig.update_layout(
            title=title,
            xaxis_tickformat=time_format,
            width=figsize[0],
            height=figsize[1],
        )

        if return_figure:
            return fig

        fig.show()

    def close(self) -> None:
        plt.close()
