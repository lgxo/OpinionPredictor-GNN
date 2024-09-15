import matplotlib.pyplot as plt
import os
import json
# from scipy.interpolate import make_interp_spline
# from scipy.ndimage.filters import gaussian_filter1d
import numpy as np


def draw_bar(height_lst, ylim=None, mode=None):
    plt.bar(
        x = [
            "Neutral",
            "Happy",
            "Angry",
            "Sad",
            "Fear",
            "Surprise"
        ],
        height=height_lst,
        color=[
            (0, 176/255, 80/255),
            (237/255, 125/255, 49/255),
            (0/255, 176/255, 240/255),
            (169/255, 209/255, 142/255),
            (47/255, 85/255, 151/255),
            (124/255, 124/255, 124/255)
        ],
        width=0.6
    )
    # plt.title("Emotion Distribution"+f"({mode})")
    # plt.xlabel("Emotion")
    # plt.ylabel("Frequency")
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.xticks(fontproperties='Times New Roman', size=55)
    plt.yticks(fontproperties='Times New Roman', size=55)

    plt.show()


def draw_distribution(channel, news_id):
    case_lst = None
    with open(os.path.join("data/materials/casestudy", f"{channel}_top.json"), "r", encoding="utf-8") as f:
        case_lst = json.load(f)
    if case_lst is None:
        exit(0)
    for case in case_lst:
        if case["news_id"] == news_id:
            references = case["references"]
            predictions = case["predictions"]
            ylim = max(references + predictions) + 0.1
            draw_bar(references, ylim, "references")
            draw_bar(predictions, ylim, "predictions")
            break


def draw_curve(step_lst, ce_lst, titel, smoothing, **kwargs):
    plt.title(titel, fontproperties='Times New Roman', size=55)
    # plt.xlabel("Steps", fontproperties='Times New Roman', loc="right", size=55, labelpad=None)
    plt.xlabel("Steps", fontproperties='Times New Roman', loc="center", size=55, labelpad=None)
    plt.ylabel("Cross-Entropy Loss", fontproperties='Times New Roman', size=55)

    plt.xlim(0, kwargs["xlim"])
    plt.ylim(kwargs["ylim_bottom"], kwargs["ylim_top"])

    plt.xticks(np.arange(0, kwargs["xlim"], kwargs["xticks"]), fontproperties='Times New Roman', size=55)

    if kwargs["yticks"] is not None:
        plt.yticks(np.arange(kwargs["ylim_bottom"], kwargs["ylim_top"], 0.005), fontproperties='Times New Roman', size=50)
    else:
        interval = (kwargs["ylim_top"] - kwargs["ylim_bottom"]) / 5
        plt.yticks(np.arange(kwargs["ylim_bottom"], kwargs["ylim_top"]+interval, interval), fontproperties='Times New Roman', size=50)
        # plt.yticks(np.arange(kwargs["ylim_bottom"], kwargs["ylim_top"], interval), fontproperties='Times New Roman', size=50)

    plt.grid(True)

    plt.plot(step_lst, ce_lst, label="CE", color="b", linewidth=3, markerfacecolor="y", markeredgecolor='k', marker='o', markersize=20)
    
    # m = make_interp_spline(step_lst, ce_lst)
    # xs = np.linspace(0, 3100, 500)
    # ys = m(xs)
    # plt.plot(xs, ys, color="r")

    # y_smooth = gaussian_filter1d(ce_lst, sigma=0.1)
    # plt.plot(step_lst, y_smooth, color="r")

    plt.show()


def draw_CE_curve(channel):
    data = None
    with open(os.path.join("data/materials/curves", f"{channel}.json")) as f:
        data = json.load(f)
    if data is None:
        exit(0)
    
    step_lst, ce_lst = zip(*[(datus[1], datus[2]) for datus in data])
    fig_title = ""
    args = {}
    if channel == "cj":
        fig_title = "Finance"
        args = {
            "xlim": 3200,
            "ylim_bottom": 1.25,
            "ylim_top": 1.32,
            "xticks": 400,
            # "yticks": 0.005,
            "yticks": None,
        }
    elif channel == "ty":
        fig_title = "Sports"
        args = {
            "xlim": 2400,
            "ylim_bottom": 1.07,
            "ylim_top": 1.14,
            "xticks": 300,
            # "yticks": 0.005,
            "yticks": None,
        }
    else:
        fig_title = "NEED TO BE SET"
    draw_curve(step_lst, ce_lst, titel=fig_title, smoothing=5, **args)


if __name__ == '__main__':
    # h_lst_reference = [0.1789, 0.1789, 0.4863, 0.0658, 0.0242, 0.0658]
    # h_lst_predict = [0.1382, 0.2292, 0.4545, 0.1178, 0.0138, 0.0465]
    # draw_bar_chart(h_lst_reference)
    # draw_bar_chart(h_lst_predict)
    # draw_bar([0.1382, 0.2292, 0.4545, 0.1178, 0.0138, 0.0465])

    # draw_CE_curve("cj")
    draw_CE_curve("ty")
    # # news_id_choosed = "comos-kyakumy5756940"
    # news_id_choosed = "comos-kyakumy5271954"    # 最多评论情感有多个
    # draw_distribution("ty", news_id_choosed)

    # from models.baselines import baseline_lst
    # with open("url.txt", "w", encoding="utf-8") as f:
    #     f.writelines([f"https://huggingface.co/{baseline.model_name_or_path}\n" for baseline in baseline_lst])

