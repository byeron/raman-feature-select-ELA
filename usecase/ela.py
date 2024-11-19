import matplotlib.pyplot as plt
import pandas as pd

from elapy import elapy as ela


def bind_state_label_with_state_no(_data, graph):
    data = _data.copy().T
    # dataのindexラベルのsuffixを削除
    data.index = [i.split(".")[0] for i in data.index]

    # 桁合わせのために最大桁数を取得
    max_len = len(bin(graph.index.max())[2:])
    state_no_bins = {}
    for s_no, b in zip(graph.loc[:, "state_no"], graph.index):
        # 初めて出現するstate_noの場合
        # 空のリストを紐付けしてバイナリ列を追加する
        if s_no not in state_no_bins:
            state_no_bins[s_no] = []

        # state noに対応するバイナリ列を追加
        b = bin(b)[2:].zfill(max_len)
        state_no_bins[s_no].append(b)

    data["binary"] = data.apply(lambda x: "".join(x.astype(str)), axis=1)
    data["state_no"] = data["binary"].map(
        lambda x: [k for k, v in state_no_bins.items() if x in v][0]
    )
    data["energy"] = data["binary"].map(lambda x: graph.loc[int(x, 2), "energy"])
    data = data[["state_no", "binary", "energy"]]
    return data


def count_label_in_each_state(_data, normalize=False, energy_threshold=None):
    data = _data.copy()
    if energy_threshold is not None:
        data = data[data["energy"] < energy_threshold]
    data = data.reset_index()

    count, ratio = count_x_in_each_y(
        data,
        x="index",
        y="state_no",
        normalize=normalize,
    )
    return count, ratio


def count_state_no_in_each_label(
    _data,
    normalize=False,
    energy_threshold=None,
):
    data = _data.copy()
    if energy_threshold is not None:
        data = data[data["energy"] < energy_threshold]
    data = data.reset_index()

    count, ratio = count_x_in_each_y(
        data,
        x="state_no",
        y="index",
        normalize=normalize,
    )
    return count, ratio


def calc_increments(data, x, y, normalize=False):
    if normalize:
        # データ数に偏りがある場合
        # データ数が少ないクラスのカウント時の重みを増やし、
        # データ数が多いクラスのカウント時の重みを減らす
        n_classes = len(data[x].unique())
        ideal = 1 / n_classes
        actual = data[x].value_counts(normalize=True)
        increments = {i: ideal / actual[i] for i in data[x].unique()}
    else:
        increments = {i: 1 for i in data[x].unique()}
    return increments


def count_x_in_each_y(
    data: pd.DataFrame, x: str = "index", y: str = "state_no", normalize: bool = False
):
    print(data.loc[:, [x, y]])
    increments = calc_increments(data, x, y, normalize)
    counter = {}
    for _x, _y in data.loc[:, [x, y]].values:
        if _y not in counter:
            counter[_y] = {}
        if _x not in counter[_y]:
            counter[_y][_x] = 0
        counter[_y][_x] += increments[_x]

    # np.float64を組み込みのfloatに変換する
    counter = {k: {k_: float(v_) for k_, v_ in v.items()} for k, v in counter.items()}
    count = pd.DataFrame(counter).T
    print(count)
    ratio = count.div(count.sum(axis=1), axis=0)
    return count, ratio


def calc_el(data):
    h, W = ela.fit_exact(data)
    acc1, acc2 = ela.calc_accuracy(h, W, data)
    graph = ela.calc_basin_graph(h, W, data)
    D = ela.calc_discon_graph(h, W, data, graph)
    freq, trans, trans2 = ela.calc_trans(data, graph)
    _ = ela.calc_trans_bm(h, W, data)

    ela.plot_local_min(data, graph)
    ela.plot_basin_graph(graph)
    ela.plot_discon_graph(D)
    ela.plot_landscape(D)
    ela.plot_trans(freq, trans, trans2)
    n, k = data.shape
    th = ela.calc_depth_threshold(n, k)
    print(f"Depth threshold: {th}")

    return graph, (acc1, acc2)


def plot_stack_bar(
    data,
    x="State No.",
    y="Ratio",
    legend_name="Label",
    output_path="fig_stack_bar.png",
):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    data.plot(kind="bar", stacked=True, ax=ax, edgecolor="white")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(title=legend_name)
    plt.tight_layout()
    fig.savefig(output_path)


def run_ela(data, energy_threshold=None, weighted_count=False):
    graph, accs = calc_el(data)
    # print(accs[0], accs[1])
    print(graph)

    bind_data = bind_state_label_with_state_no(data, graph)

    count, ratio = count_state_no_in_each_label(
        bind_data, normalize=weighted_count, energy_threshold=energy_threshold
    )
    count2, ratio2 = count_label_in_each_state(
        bind_data, normalize=weighted_count, energy_threshold=energy_threshold
    )

    plot_stack_bar(
        ratio, x="Label", legend_name="State no.", output_path="fig_ratio_state_no.png"
    )
    plot_stack_bar(
        ratio2, x="State no.", legend_name="Label", output_path="fig_ratio_label.png"
    )
    plot_stack_bar(
        count, x="Label", legend_name="State no.", output_path="fig_count_state_no.png"
    )
    plot_stack_bar(
        count2, x="State no.", legend_name="Label", output_path="fig_count_label.png"
    )
