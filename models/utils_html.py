from bs4 import BeautifulSoup


def get_paragraphs(soup, cut_len: int) -> list[str]:
    contents = []
    current_heading = ""  # 用于保存最近的标题
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # Add the appropriate number of # characters before the heading
            current_heading = (
                "#" * (int(tag.name[1]) + 2)
                + " "
                + " ".join(tag.get_text().strip())
                + "\n"
            )
        else:
            # Add the current heading to the beginning of the paragraph
            curr_para = current_heading + " ".join(tag.get_text().strip())
            curr_para = curr_para[:cut_len]
            contents.append(curr_para)
    return contents


def clean_html(webpage, remove_links=False):
    soup = BeautifulSoup(webpage, features="html.parser")
    # pprint(soup.get_text())

    for script in soup.find_all(["script", "style", "meta"]):
        script.decompose()  # Remove these tags

    # 查找并删除所有 <a> 标签。删除网页链接
    # Update: 有时候 <a> 标签是带有信息的，暂时不删除
    if remove_links:
        for link in soup.find_all("a"):
            link.unwrap()  # decompose 方法会从文档树中移除该元素

    # 删除页眉
    # for header in soup.find_all(['header', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
    #     header.decompose()

    # 删除页脚
    for footer in soup.find_all("footer"):
        footer.decompose()

    # 删除所有按钮的文字
    for button in soup.find_all("button"):
        button.decompose()

    # Find all text nodes and replace escaped Unicode characters
    for text_element in soup.find_all(String=True):
        if "\\u" in text_element or "\\U" in text_element:
            # Decode the escaped unicode characters
            decoded_text = bytes(text_element, "utf-8").decode("unicode_escape")
            text_element.replace_with(decoded_text)

    return soup


def get_content(soup):
    content = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # Add the appropriate number of # characters before the heading
            markdown_heading = "#" * int(tag.name[1]) + " " + tag.get_text()
            content.append(markdown_heading)
        else:
            content.append(tag.get_text())
    return content


def check_empty_table(table_str: str) -> bool:
    table_str = (
        table_str.replace("|", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("\n", "")
        .strip()
    )
    table_str = "".join(table_str.split())
    if table_str == "":
        return True
    return False


def get_tables(soup, html_name: str) -> list[str]:
    markdown_tables = []

    # Find all table tags
    tables = soup.find_all("table")
    for table in tables:
        markdown_table = []
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["th", "td"])
            # Extract text from each cell and join them with a pipe
            formatted_row = (
                "| " + " | ".join(cell.get_text(strip=True) for cell in cells) + " |"
            )
            markdown_table.append(formatted_row)
        # Add a separator after the header (assumes the first row is header)
        if markdown_table:
            header_sep = (
                "| "
                + " | ".join("---" for _ in markdown_table[0].split("|")[1:-1])
                + " |"
            )
            markdown_table.insert(1, header_sep)
        markdown_table_str = "\n".join(markdown_table)
        if check_empty_table(markdown_table_str):
            continue
        markdown_table_str = f"Page name: {html_name}\n{markdown_table_str}"
        markdown_tables.append(markdown_table_str)
    return markdown_tables
