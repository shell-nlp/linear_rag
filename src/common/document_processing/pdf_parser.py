import copy
import io
import re
from typing import Any, Iterable, List, Optional

import fitz
import pandas as pd
import pdfplumber
import PIL
import PyPDF2
from langchain_core.documents import Document
from langchain_text_splitters.character import (
    RecursiveCharacterTextSplitter,
)

from loguru import logger

from src.common.models import ParsedDocument


def _split_text_with_regex_from_end(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            r"\.\s|\!\s|\?\s",
            r"；|;\s",
            r"，|,\s",
        ]
        self._is_separator_regex = is_separator_regex

    def under_non_alpha_ratio(self, text: str, threshold: float = 0.5):
        """Checks if the proportion of non-alpha characters in the text snippet exceeds a given
        threshold. This helps prevent text like "-----------BREAK---------" from being tagged
        as a title or narrative text. The ratio does not count spaces.

        Parameters
        ----------
        text
            The input string to test
        threshold
            If the proportion of non-alpha characters exceeds this threshold, the function
            returns False
        """
        if len(text) == 0:
            return False

        alpha_count = len([char for char in text if char.strip() and char.isalpha()])
        total_count = len([char for char in text if char.strip()])
        try:
            ratio = alpha_count / total_count
            return ratio < threshold
        except:
            return False

    def is_possible_title(
        self,
        text: str,
        title_max_word_length: int = 20,
        non_alpha_threshold: float = 0.5,
    ) -> bool:
        """Checks to see if the text passes all of the checks for a valid title.

        Parameters
        ----------
        text
            The input text to check
        title_max_word_length
            The maximum number of words a title can contain
        non_alpha_threshold
            The minimum number of alpha characters the text needs to be considered a title
        """

        # 文本长度为0的话，肯定不是title
        if len(text) == 0:
            # print("Not a title. Text is empty.")
            return (False, 0)

        # 文本中有标点符号，就不是title
        ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
        ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
        if ENDS_IN_PUNCT_RE.search(text) is not None:
            return (False, 0)

        # 文本长度不能超过设定值，默认20
        # NOTE(robinson) - splitting on spaces here instead of word tokenizing because it
        # is less expensive and actual tokenization doesn't add much value for the length check
        if len(text) > title_max_word_length:
            return (False, 0)

        # 文本中数字的占比不能太高，否则不是title
        if self.under_non_alpha_ratio(text, threshold=non_alpha_threshold):
            return (False, 0)

        # NOTE(robinson) - Prevent flagging salutations like "To My Dearest Friends," as titles
        if text.endswith((",", ".", "，", "。")):
            return (False, 0)

        if text.isnumeric():
            # print(f"Not a title. Text is all numeric:\n\n{text}")  # type: ignore
            return (False, 0)

        # 开头的字符内应该有数字，默认5个字符内
        if len(text) < 5:
            text_5 = text
        else:
            text_5 = text[:5]
        alpha_in_text_5 = sum(list(map(lambda x: x.isnumeric(), list(text_5))))
        if not alpha_in_text_5:
            return (False, 0)

        return (True, 0)

    def split_documents3(
        self, documents: Iterable[Document], chunk_size: int = 20
    ) -> List[Document]:
        """
        处理Document对象列表，合并跨页标题内容
        输入: List[Document] (每个Document包含content和metadata)
        输出: 按标题分块的新Document列表
        """
        chunks = []
        current_chunk_content = []
        current_chunk_meta = None

        # 按页码排序确保顺序（假设metadata中有page_num）
        sorted_docs = sorted(documents, key=lambda x: x.metadata.get("pages_number", 0))
        for doc in sorted_docs:
            lines = doc.page_content.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                is_title, level = self.is_possible_title(line)
                if is_title:
                    # 保存当前chunk
                    if current_chunk_content:
                        # === 修改点1：统一添加ori_text字段（无论是否为标题块） ===
                        current_chunk_meta["ori_text"] = "\n".join(
                            line
                            for line in current_chunk_content
                            if not line.startswith("#  ")
                        )
                        # 创建新Document对象
                        chunks.append(
                            Document(
                                page_content="\n".join(current_chunk_content),
                                metadata=current_chunk_meta,
                            )
                        )

                    # 开始新chunk
                    current_chunk_content = [f"{'#'} {line}"]
                    current_chunk_meta = {
                        "title": line,
                        "pages_number": doc.metadata.get("pages_number"),
                        # 'end_page': doc.metadata.get('page'),
                        "content_table": [],
                        "content_image": [],
                    }
                else:
                    # 添加到当前chunk
                    if current_chunk_content:
                        current_chunk_content.append(line)
                        if current_chunk_meta:
                            current_chunk_meta["pages_number"] = doc.metadata.get(
                                "pages_number"
                            )
                    else:
                        # 文档开头的无标题内容
                        current_chunk_content = [line]
                        current_chunk_meta = {
                            "title": "无标题内容",
                            "pages_number": doc.metadata.get("pages_number"),
                            # 'end_page': doc.metadata.get('page'),
                            "content_table": [],
                            "content_image": [],
                        }
        # 添加最后一个chunk
        if current_chunk_content:
            # === 修改点2：统一添加ori_text字段（无论是否为标题块） ===
            current_chunk_meta["ori_text"] = "\n".join(
                line for line in current_chunk_content if not line.startswith("#  ")
            )

            chunks.append(
                Document(
                    page_content="\n".join(current_chunk_content),
                    metadata=current_chunk_meta,
                )
            )

        if len(chunks) > 100:
            chunks = self.merge_chunks_simple(chunks=chunks, max_length=2000)

        return chunks

    #     pass

    def merge_chunks_simple(
        self, chunks: List[Document], max_length: int = 500
    ) -> List[Document]:
        """
        合并过小的chunks直到达到指定长度
        chunks: 已划分的Document列表
        max_length: 最小目标长度（字符数）
        返回: 合并后的Document列表
        """
        if not chunks:
            return []

        merged_chunks = []
        current_content = []
        current_meta = None
        temp_contents = []  # 用于收集原始内容

        for chunk in chunks:
            # 获取chunk的原始文本（不含markdown标记）
            page_content = chunk.page_content

            # 计算当前内容长度（如果已经有内容）
            current_len = len("".join(temp_contents)) if temp_contents else 0
            chunk_len = len(page_content)

            # 如果是第一个chunk或者当前合并块还很小
            if current_meta is None or current_len + chunk_len <= max_length:
                if current_meta is None:
                    # 开始一个新的合并块，记录第一个chunk的元数据
                    current_meta = chunk.metadata.copy()
                    current_meta["merged_from"] = [current_meta.get("title", "无标题")]
                else:
                    # 添加到当前合并块的来源列表
                    current_meta["merged_from"].append(
                        chunk.metadata.get("title", "无标题")
                    )

                # 收集内容
                temp_contents.append(page_content)
                current_content.append(page_content)
            else:
                # 当前合并块已达标，保存它
                if current_content:
                    merged_chunks.append(
                        Document(
                            page_content="\n".join(current_content),
                            metadata={
                                "title": current_meta.get(
                                    "title", "无标题"
                                ),  # 使用第一个chunk的title
                                "pages_number": current_meta.get(
                                    "pages_number"
                                ),  # 使用第一个chunk的页码
                                "content_table": [],
                                "content_image": [],
                                "ori_text": "\n".join(temp_contents),
                                "is_merged": True,
                                "merged_from": current_meta["merged_from"],
                            },
                        )
                    )

                # 开始新的合并块，以当前chunk为起点
                current_content = [page_content]
                current_meta = chunk.metadata.copy()
                current_meta["merged_from"] = [current_meta.get("title", "无标题")]
                temp_contents = [page_content]

        # 处理最后一个合并块
        if current_content:
            merged_chunks.append(
                Document(
                    page_content="\n".join(current_content),
                    metadata={
                        "title": current_meta.get("title", "无标题"),
                        "pages_number": current_meta.get("pages_number"),
                        "content_table": [],
                        "content_image": [],
                        "ori_text": "\n".join(temp_contents),
                        "is_merged": True,
                        "merged_from": current_meta["merged_from"],
                    },
                )
            )

        return merged_chunks

    def split_documents2(
        self, documents: Iterable[Document], chunk_size: int = 20
    ) -> List[Document]:
        # doc 表示每一页 对每一页再切片
        docs = []
        for doc in documents:
            content_table = doc.metadata["content_table"]
            content_image = doc.metadata["content_image"]
            full_text = doc.page_content
            # ----------------------------------------------------------------
            pattern_table = r"\*\*【表格\d+】\*\*"
            # 使用re.findall找到所有匹配的文件名
            table_matchers = re.findall(pattern_table, full_text)
            # 将表格序号重排
            for table_i, table_mark in enumerate(table_matchers):
                full_text = full_text.replace(table_mark, f"**【表格{table_i}】**")
            # 开始切片
            pattern = "(" + "|".join(map(re.escape, self._separators)) + ")"
            sentences = re.split(pattern, full_text)
            pattern_table_id = r"\*\*【表格(\d+)】\*\*"
            for i in range(0, len(sentences), chunk_size):
                text = "".join(sentences[i : i + chunk_size])
                table_matchers_text = re.findall(pattern_table, text)
                for table_tag in table_matchers_text:
                    table_matchers_id = re.findall(pattern_table_id, table_tag)
                    text = text.replace(
                        table_tag, content_table[int(table_matchers_id[0])]
                    )
                doc_ = Document(
                    page_content=text,
                    metadata={
                        "content_pages_number": doc.metadata["content_pages_number"],
                        # "content_table": content_table,
                        "content_image": content_image,
                    },
                )
                docs.append(doc_)
        return docs

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip() != ""
        ]


text_splitter = ChineseRecursiveTextSplitter(
    chunk_size=600,
    chunk_overlap=0,
)


def get_max_depth(outline, current_depth=1):
    """
    递归地遍历目录结构，确定最大深度。

    :param outline: 目录项列表
    :param current_depth: 当前深度
    :return: 最大深度
    """
    max_depth = current_depth
    for item in outline:
        if isinstance(item, list):
            max_depth = max(max_depth, get_max_depth(item, current_depth + 1))
    return max_depth


def detect_pdf_structure(file_bytes):
    # new way
    # doc = fitz.Document(stream=file_bytes)
    # title_info = doc.get_toc()
    # title_level_list = [i[0] for i in title_info]

    # TextIO and BinaryIO.
    stream = io.BytesIO(file_bytes)
    reader = PyPDF2.PdfReader(stream)
    outlines = reader.outline

    toc = list(outlines)
    if not toc:
        return {"split_strategy": "chunksplit"}

    max_depth = get_max_depth(toc)
    return {
        "split_strategy": "titlesplit",
        "max_title_level": max_depth,
    }


def find_nearest_line(text_lines, chars, anchor_y):
    nearest_line_index = None
    nearest_line_y = None
    min_distance = float("inf")

    for index, line in enumerate(text_lines):
        line_chars = [char for char in chars if char["text"] in line]
        if line_chars:
            line_y = sum(char["top"] for char in line_chars) / len(line_chars)
            distance = abs(line_y - anchor_y)
            if distance < min_distance:
                nearest_line_index = index
                nearest_line_y = line_y
                min_distance = distance

    return nearest_line_index, nearest_line_y


def extract_parent_title(title_info, sub_title, cur_level):
    """
    从目录信息中提取某个标题的父标题。

    :param title_info: list，从 fitz.Document.get_toc() 获得的目录信息，包含 [level, title, page]。
    :param sub_title: str，目标标题名称。
    :param cur_level: int，目标标题级别。
    :return: list，父标题及其级别的列表，如 [(1, "密云水库高水位运行工程安全保障方案")]。
    """
    parent_titles = []

    # 找到目标标题的索引位置
    target_index = None
    for i, (level, title, page) in enumerate(title_info):
        if title == sub_title and level == cur_level:
            target_index = i
            break

    # 如果没有找到目标标题，返回空列表
    if target_index is None:
        return parent_titles

    # 从目标标题向前查找父标题
    for i in range(target_index - 1, -1, -1):
        level, title, _ = title_info[i]
        if level < cur_level:
            parent_titles.append((level, title))
            cur_level = level  # 更新当前级别为找到的父标题级别

    # 返回从最高层级到最低层级的父标题列表
    return parent_titles[::-1]


def levenshtein_distance(s1, s2):
    """计算编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def extract_toc_from_fitz(title_info, level):
    """
    从PDF目录大纲中提取指定级别的标题。

    :param title_info: 从fitz.Document.get_toc()获得的目录信息。
    :param level: 要提取的标题级别（整数）。
    :return: 包含指定级别标题的列表，每个标题是一个字典，包含标题文本和页码,和标题等级。
    """
    return [
        (entry[1].strip(), entry[2], entry[0])
        for entry in title_info
        if entry[0] <= level
    ]


def insert_mark_near_position(text_lines, chars, bbox, mark):
    # 找到最接近边界框顶部的文本行
    nearest_line_index, nearest_line_y = find_nearest_line(text_lines, chars, bbox[1])

    if nearest_line_index is not None:
        # 在最接近的文本行插入标记
        text_lines[nearest_line_index] += " " + mark

    return "\n".join(text_lines)


def convert_title_with_paragraph_breaks(text):
    # 正则解释：
    # \s*      : 匹配数字前可能存在的空格（比如“标识”和“1.”之间的空格）
    # (\d+\.)  : 捕获组，匹配1个或多个数字紧跟一个点（如 1. 2. 10.）
    # (?!\d)   : 负向先行断言，确保点的后面不是数字（防止把金额 1.5万元 拆行）

    # 替换规则：将匹配到的内容替换为“换行符 + 原来的数字和点”
    formatted_text = re.sub(r"\s*(\d+\.)(?!\d)", r"\n\1", text)

    # 确保首尾没有多余的空格，并在最后加上一个换行符
    return formatted_text.strip() + "\n"


class PDFParser:
    """PDF 文档解析器，只处理字节内容，不负责文件持久化。"""

    def __init__(
        self,
        bucket_name: str,
        object_key: str,
        file_id: str,
    ):
        """保存文件来源元数据，实际内容由 get_chunk 直接接收。"""

        self.bucket_name = bucket_name
        self.object_key = object_key
        self.file_id = file_id

    def parse(
        self,
        file_bytes: bytes,
        bucket_name: str | None = None,
        object_key: str | None = None,
        file_id: str | None = None,
    ) -> List[ParsedDocument]:
        """实现 DocumentParser 端口，解析上传字节并返回统一文档模型。"""

        if bucket_name is not None:
            self.bucket_name = bucket_name
        if object_key is not None:
            self.object_key = object_key
        if file_id is not None:
            self.file_id = file_id
        return self.get_chunk(file_bytes)

    def get_chunk(self, file_bytes: bytes) -> List[ParsedDocument]:
        """执行 PDF 结构识别、内容抽取和切片。"""

        if self.object_key:
            file_id = self.file_id
            logger.info(f"解析文件: {self.bucket_name}/{self.object_key}")
            use_table = False
            use_image = False
            logger.info(f"是否使用表格：{use_table}")
            logger.info(f"是否使用图片：{use_image}")
            # bucket_name, file_name = self.get_url_info()
            bucket_name = self.bucket_name
            file_name = self.object_key.split("/")[-1]
            bytes_io = io.BytesIO(file_bytes)
            structure_info = detect_pdf_structure(file_bytes=file_bytes)

            page_data = []
            pdf_lens = None
            with pdfplumber.open(bytes_io) as pdf:
                pdf_lens = len(pdf.pages)
                logger.debug(f"{self.object_key} PDF页数：{pdf_lens}")
                for page_number, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text()
                    except Exception as e:
                        logger.warning(
                            f"对象: {self.object_key} 第{page_number}页抽取文本异常 {e}，已跳过"
                        )
                        continue
                    page_image_list = []
                    if use_image:
                        # ----------------------------
                        images = page.images
                        for image_number, image in enumerate(images):
                            try:
                                image_data_bin = image["stream"].get_data()
                                image_data = io.BytesIO(image_data_bin)
                                try:
                                    image_data_ = copy.deepcopy(image_data)
                                    pil_image = PIL.Image.open(image_data_)
                                    pil_image.verify()
                                except PIL.UnidentifiedImageError as e:
                                    logger.warning("图片验证失败，已跳过")
                                    continue

                                text_lines = text.split("\n")
                                bbox = (
                                    image["x0"],
                                    image["top"],
                                    image["x1"],
                                    image["bottom"],
                                )
                                image_name = f"image_{bucket_name}_{file_id}_{page_number}_{image_number}.png"
                                page_image_list.append(f"{bucket_name}/" + image_name)
                                text_with_image_mark = insert_mark_near_position(
                                    text_lines,
                                    page.chars,
                                    bbox,
                                    image_name,
                                )
                                text = text_with_image_mark

                            except IndexError as e:
                                logger.warning("图片上传错误，已经跳过！")
                                continue
                    page_md_list = []
                    if use_table:
                        # ----------------------------

                        tables_list = (
                            page.extract_tables()
                        )  #  [[['事件名称', '事件时间', '事件概述', None, None], [...
                        # print(text)
                        # print("######################")
                        tables = page.find_tables()
                        page_table_idx = []
                        if tables:
                            table_id = 0
                            for i, table in enumerate(tables):
                                try:
                                    table_area = page.within_bbox(table.bbox)
                                    table_inner_text = table_area.extract_text()
                                    idx = text.find(table_inner_text)
                                    if idx != -1:
                                        if len(table_inner_text.split("\n")) < 2:
                                            continue
                                        text = text.replace(
                                            table_inner_text, f"**【表格{table_id}】**"
                                        )
                                        table_id += 1
                                        page_table_idx.append(i)
                                except ValueError as e:
                                    logger.warning("表格解析发生值错误,已跳过")
                        # print(text)
                        # --------------------开始处理表格--------------------
                        for i in page_table_idx:
                            md_list = []
                            for idx, row_list in enumerate(tables_list[i]):
                                if idx == 0:
                                    header = [i for i in row_list if i]
                                    header_len = len(header)  # 记录 表头的 长度
                                    # print(header)
                                    md_list.append(header)
                                    continue
                                sub_row = [i for i in row_list if i]
                                frist_is_none = row_list[0] == None
                                if (
                                    len(sub_row) == header_len
                                ):  # 如果 子行的长度 等于 表头的长度 则插入
                                    md_list.append(sub_row)
                                elif (
                                    len(sub_row) < header_len
                                    and idx != len(tables_list[i]) - 1
                                ):  # 长度不足 且 不等与 最后一行
                                    md_list[-1][-1] += "".join(sub_row)
                                elif (
                                    len(sub_row) < header_len
                                    and idx == len(tables_list[i]) - 1
                                    and frist_is_none
                                ):
                                    md_list[-1][-1] += "".join(sub_row)
                                elif (
                                    len(sub_row) < header_len
                                    and idx == len(tables_list[i]) - 1
                                    and not frist_is_none
                                ):
                                    sub_row.extend([""] * (header_len - len(sub_row)))
                                    md_list.append(sub_row)
                            # print()
                            # print(md_list, len(md_list))
                            columns = md_list[0]
                            columns = [i.replace("\n", "") for i in columns]
                            df = pd.DataFrame(md_list[1:], columns=columns).map(
                                lambda x: x.replace("\n", "")
                            )
                            page_md = df.to_markdown(index=False)
                            page_md_list.append(page_md)
                    page_data.append(
                        {
                            "text": text,
                            "pages_number": page_number,
                            "content_table": page_md_list,
                            "content_image": page_image_list,
                        }
                    )
            docs = [
                Document(page_content=page_info.get("text", ""), metadata=page_info)
                for page_info in page_data
            ]

            logger.debug(f"{self.object_key} 解析完成后的页数：{len(docs)}")
            logger.debug("开始切片！")
            split_strategy = structure_info["split_strategy"]
            logger.info(f"split_strategy: {split_strategy}")
            if split_strategy == "chunksplit":
                docs = text_splitter.split_documents3(docs)
                # for idx,doc in enumerate(docs[52:62]):
                #     logger.warning(f"idx:{idx},doc:{doc}")
                #     print("\n")

            elif split_strategy == "titlesplit":
                chunksplit_docs = text_splitter.split_documents3(docs)
                title_level = structure_info["max_title_level"]
                logger.info(f"title_level: {title_level}")
                doc = fitz.Document(stream=file_bytes)
                title_info = doc.get_toc()
                toc = extract_toc_from_fitz(title_info, level=title_level)
                # --- 对 toc 进行校验处理----
                toc_copy = copy.deepcopy(toc)
                toc = []
                for idx in range(len(toc_copy)):
                    # if idx == 0:
                    #     continue
                    # if toc_copy[idx][1] < toc_copy[idx - 1][1]:
                    #     continue
                    # else:
                    toc.append(toc_copy[idx])
                # --- 对 toc 进行校验处理----
                # logger.info(f"toc: {toc}  len: {len(toc)}")
                # ---------- TODO 为了修复丢失的开始的内容 修改 toc  ----------
                toc.insert(0, ("$#", 1, 1))

                logger.info(f"toc: {toc}  len: {len(toc)}")
                # ---------- TODO 为了修复丢失的开始的内容 修改 toc  ----------

                title_map = {}
                title_page_map = {}
                for i in range(len(toc)):
                    title, page, level = toc[i]
                    if i < len(toc) - 1:
                        next_page = toc[i + 1][1]
                        # page 是从 1 开始的 docs列表是从 0 开始的
                        # title_page_map[title] = f"{page} -- {next_page}"
                        title_page_map[title] = (page, next_page)
                        title_map[title] = docs[
                            page - 1 : next_page
                        ]  # TODO 是否要 +1 因为next_page 也需要
                        # print(
                        #     f"标题： {title}  {page}-{next_page}  页数：{len(title_map[title])}"
                        # )
                    else:
                        # title_page_map[title] = f"{page} -- 最后一页"
                        title_page_map[title] = (page, pdf_lens)
                        title_map[title] = docs[page - 1 :]
                        # print(
                        #     f"标题： {title}  {page} -- {len(docs)}   页数：{len(title_map[title])}"
                        # )
                # 对 title_map 进行融合
                title_docs = []
                logger.info(
                    f"title_page_map： {title_page_map}  \nlen: {len(title_map)}"
                )
                i = 0
                for idx, (title, docs) in enumerate(title_map.items()):
                    i += 1
                    text = ""
                    content_table = []
                    content_image = []
                    if docs:
                        for doc in docs:
                            content_table.extend(doc.metadata["content_table"])
                            content_image.extend(doc.metadata["content_image"])
                            text += doc.page_content
                        if use_table:
                            # -------------- 对融合后的多页进行表格重标记 --------------
                            # 现阶段表格还是一一对应的
                            pattern_table = r"\*\*【表格\d+】\*\*"
                            # 使用re.findall找到所有匹配的文件名
                            table_matchers = re.findall(pattern_table, text)
                            logger.debug(
                                f"table_matchers len {len(table_matchers)}  标题{toc[idx]}"
                            )
                            if len(table_matchers) >= 50:
                                for table_i, table_mark in enumerate(table_matchers):
                                    text = text.replace(table_mark, "")
                            else:  # 对表格序号进行重新标记
                                for table_i, table_mark in enumerate(table_matchers):
                                    text = text.replace(
                                        table_mark, f"**【表格{table_i}】**"
                                    )

                        # -------------- 对融合后的多页进行表格重标记 --------------
                        # 寻找真实的标题位置
                        text_splits = text.split("\n")
                        # 原始的 text_splits, 用来记录不添加标题的内容
                        ori_text_splits = copy.deepcopy(text_splits)
                        title_start_idx = 0
                        title_start_flag = False

                        title_end_idx = 100000
                        title_end_flag = False
                        for text_idx in range(len(text_splits)):
                            if title_start_flag and title_end_flag:
                                break
                            if not title_start_flag:
                                distance_start = levenshtein_distance(
                                    s1=text_splits[text_idx], s2=toc[idx][0]
                                )
                                if (
                                    distance_start <= 2 or toc[idx][0] == "$#"
                                ):  # 是标题 记录位置
                                    # 添加父标题   [(1, "密云水库高水位运行工程安全保障方案")]
                                    parent_title_list = extract_parent_title(
                                        title_info=title_info,
                                        sub_title=toc[idx][0],
                                        cur_level=toc[idx][-1],
                                    )
                                    parent_title_str = ""
                                    for parent_title in parent_title_list:
                                        parent_title_str += (
                                            "#" * int(parent_title[0])
                                            + " "
                                            + parent_title[1]
                                            + "\n"
                                        )
                                    # print(
                                    #     f"子标题 {text_splits[text_idx]} 父标题：{parent_title_str}"
                                    # )
                                    # print("-------")
                                    text_splits[text_idx] = parent_title_str + (
                                        "#" * int(toc[idx][-1])
                                        + " "
                                        + text_splits[text_idx]
                                        + "\n"
                                    )
                                    title_start_idx = text_idx
                                    title_start_flag = True
                                    continue
                            if not title_end_flag:
                                if idx + 1 == len(title_map.items()):
                                    continue
                                distance_end = levenshtein_distance(
                                    s1=text_splits[text_idx], s2=toc[idx + 1][0]
                                )
                                if distance_end <= 2:  # 是标题 记录位置
                                    title_end_idx = text_idx
                                    title_end_flag = True
                                    continue
                        text = "".join(text_splits[title_start_idx:title_end_idx])
                        ori_text = "".join(
                            ori_text_splits[title_start_idx:title_end_idx]
                        )
                        ori_text = convert_title_with_paragraph_breaks(ori_text)
                        # print("原始标题：",ori_text)
                        # print("原始标题1：",ori_text1)

                        # logger.debug(f"位置：{title_start_idx} | {title_end_idx}")
                        # logger.debug(f"标题：{title} | {toc[idx]}")
                        # logger.warning(f"标题片：\n{text}")

                        # logger.debug("页内容拼凑完成！")
                        new_content_image = []
                        if use_image:
                            pattern_image = r"image_[\w-]+_\d+_\d+\.png"
                            images_anchor = re.findall(pattern_image, text)
                            new_content_image = [
                                f"{bucket_name}/" + image_name
                                for image_name in images_anchor
                            ]
                        new_content_table = []
                        if use_table:
                            # 使用re.findall找到所有匹配的文件名
                            pattern_table_id = r"\*\*【表格(\d+)】\*\*"
                            table_id_matchers = re.findall(pattern_table_id, text)
                            for table_i, table_id in enumerate(table_id_matchers):
                                new_content_table.append(content_table[int(table_id)])
                        # logger.debug("表格替换完成！")

                        title_doc = Document(
                            page_content=text,
                            metadata={
                                "pages_number": docs[0].metadata["pages_number"],
                                "content_table": new_content_table,
                                "content_image": new_content_image,
                                "ori_text": ori_text,
                            },
                        )
                        if len(title_doc.page_content) <= 30:
                            continue
                        title_docs.append(title_doc)
                docs = title_docs or chunksplit_docs
                # print(docs[0]) # TODO 缺少
            logger.info(f"最终片数： {len(docs)}")
            if len(docs) == 0:
                raise Exception(
                    "该文件疑似为扫描件，不含任何字符,文件解析失败",
                )

            # shared_tenant_id_list = requests.shared_tenant_id_list
            # shared_tenant_id_list.append(
            #     requests.tenant_id
            # )  # 为shared_tenant_id_list 添加自身的tenant_id
            for segment_id, doc in enumerate(docs, start=1):  # 切片id 从 1 开始，
                doc.metadata.update(
                    {
                        "file_name": file_name,
                        "file_id": file_id,
                        "segment_id": segment_id,
                        "state": True,
                        "bucket_name": self.bucket_name,
                        "file_path": self.object_key,
                        # "tenant_id": requests.tenant_id,
                        # "shared_tenant_id_list": shared_tenant_id_list,
                    }
                )
            logger.info(f"file_name: {file_name}  file_id: {file_id}")
            return [
                ParsedDocument(
                    text=doc.page_content,
                    metadata=dict(doc.metadata),
                )
                for doc in docs
            ]

