import re
from typing import List, Set


# clear unicode and redundant "\n"
""" space, tab, and line break """
def TextFilter(text):
    # text = text.strip()
    # filter characters that are not printable
    text_without_unicode = "".join(x for x in text if x=="\n" or x.isprintable())
    # filter redundant "\n"
    text_splited = text_without_unicode.split("\n")
    # filter blank sub-string
    text_without_blank = []
    for text in text_splited:
        blank_removed_text = re.sub(r"\s+", "", text)
        if blank_removed_text:
            text_without_blank.append(blank_removed_text)
    # text_without_blank = [text.strip() for text in text_splited if text.strip()!=""]    
    text_filtered = "\n".join(text_without_blank)
    # text_filtered = re.sub("\\n+", "\\n", text_without_unicode)
    return text_filtered


# clear unicode and redundant "\n"
def TitleFilter(title):
    return TextFilter(title)


# clear unicode and redundant "\n"
def NewsFilter(news, media):
    news_without_unicode = TextFilter(news)
    paragraph_lst = news_without_unicode.split("\n")    # 根据'\n'划分segment

    # 过滤无关段落（作者，来源），且相关段落的长度都小于10，只考察前2个段落
    unrelatedAhead_cleared = False
    for offset in range(1, -1, -1):
        if not unrelatedAhead_cleared:
            if len(paragraph_lst) > offset and len(paragraph_lst[offset]) < 10:
                if paragraph_lst[offset].find(media) != -1:
                    paragraph_lst = paragraph_lst[offset+1:]
                    unrelatedAhead_cleared = True
                    break
                for keywords in ["作者", "来源", "原标题"]:
                    if paragraph_lst[offset].startswith(keywords):
                        paragraph_lst = paragraph_lst[offset+1:]
                        unrelatedAhead_cleared = True
                        break

    # 过滤无关段落（文章来源，责任编辑等），只考察后5个段落
    unrelatedBhind_cleared = False
    for offset in range(-5, 0):
        # 如果还未处理过
        if not unrelatedBhind_cleared:
            if len(paragraph_lst) > abs(offset):
                for keywords in ['责任编辑', '来源：', '声明：新浪', 'SF', '参考资料：', '海量资讯', '原标题', '（原标题', '[原标题',
                        '资料来源：', '（文章来源：', '封面图片来源：', '（来源：', '图片来源：', '本文来源：', '（素材来源：',
                        '视频来源：', '部分来源：', '点击进入专题：', '#图片来源', '责编', '新浪声明', '主编：', '栏目主编']:
                    if paragraph_lst[offset].startswith(keywords):
                        paragraph_lst = paragraph_lst[:offset]
                        unrelatedBhind_cleared = True
                        break

    return "\n".join(paragraph_lst)


# clear unicode and redundant "\n", and merge into one sentence
def CommentFilter(comment):
    comment_without_unicode = TextFilter(comment)
    # copy
    # comment_content = re.sub(r"\[.*\]", '', comment)  # 去除表情符号
    # line_break_char = ['\n', '\t', '\r']
    # for char in line_break_char:
    #     comment_without_unicode = comment_without_unicode.replace(char, '')
    # comment_filtered = re.sub(r"\s+", "", comment_without_unicode)  # 合并连续空格

    # drop out \n 
    comment_filtered = "".join(comment_without_unicode.split("\n"))
    return comment_filtered


# load stopwords
def load_stopwords(stopwords_path):
    stopwords_set = set()
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords_set.add(line.strip())
    return stopwords_set


# clear stopwords
def stopwordsFilter(word_list: List[str], stopwords_set: Set[str]) -> List[str]:
    return [word for word in word_list if word not in stopwords_set]
