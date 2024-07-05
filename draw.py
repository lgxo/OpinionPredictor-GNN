import matplotlib.pyplot as plt


def draw_bar_chart(height):
    # color_lst = ["green", "orange", "blue", ""]
    plt.bar(
        x = [
            "neutral",
            "happy",
            "angry",
            "sad",
            "fear",
            "surprise"
        ],
        height=height
    )
    plt.show()

if __name__ == '__main__':
    h_lst_reference = [0.1789, 0.1789, 0.4863, 0.0658, 0.0242, 0.0658]
    h_lst_predict = [0.1382, 0.2292, 0.4545, 0.1178, 0.0138, 0.0465]
    draw_bar_chart(h_lst_reference)
    draw_bar_chart(h_lst_predict)
