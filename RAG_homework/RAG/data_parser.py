import re
from pathlib import Path

import pandas as pd
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode



def preprocess_latex(content):
    """
    Предобрабатывает текст LaTeX:
    - Удаляет лишние \n и пробелы.
    - Убирает команды \includegraphics, \centering и содержимое окружений figure.
    - Убирает комментарии.
    - Форматирует формулы.

    :param content: Строка с содержимым LaTeX.
    :return: Очищенный текст.
    """
    # Удаление комментариев (все, что начинается с % до конца строки)
    content = re.sub(r"(?<!\\)%.*", "", content)
    
    # Удаление лишних \n (оставляя только один между абзацами)
    content = re.sub(r"\n\s*\n", "\n", content)
    content = re.sub(r"\s*\n\s*", " ", content)  # Превращение \n в пробелы внутри строки
    
    # Удаление лишних пробелов
    content = re.sub(r"\s{2,}", " ", content)  # Множественные пробелы заменяются на один
    content = content.strip()
    
    # Удаление команд \includegraphics
    content = re.sub(r"\\includegraphics(\[.*?\])?{.*?}", "", content)
    
    # Удаление команд \centering
    content = re.sub(r"\\centering", "", content)
    
    # Удаление содержимого окружений figure
    content = re.sub(r"\\begin{figure}.*?\\end{figure}", "", content, flags=re.DOTALL)
    
    # Обработка формул в $...$ и \(...\), убираем лишние пробелы внутри формул
    # content = re.sub(r"\$(.*?)\$", lambda m: f"${m.group(1).strip()}$", content)
    # content = re.sub(r"\\\((.*?)\\\)", lambda m: f"\\({m.group(1).strip()}\\)", content)
    

    content = re.sub(r"\$.*?\$", "", content)  # Inline формулы
    content = re.sub(r"\\\[.*?\\\]", "", content, flags=re.DOTALL)  # Display формулы
    content = re.sub(r"\\\(.*?\\\)", "", content)  # Parenthesized формулы

    # Обработка формул в окружении \[...\]
    content = re.sub(r"\\\[(.*?)\\\]", lambda m: f"\\[{m.group(1).strip()}\\]", content)
    content = re.sub(r"\\[a-zA-Z]+(\{.*?\})?", "", content)
    # Удаление лишних пробелов в командах (например, \textbf{  example } -> \textbf{example})
    content = re.sub(r"\\([a-zA-Z]+)\s*{(.*?)}", lambda m: f"\\{m.group(1)}{{{m.group(2).strip()}}}", content)
    content = re.sub(r"{.*?}", "", content)
    content = re.sub(r"\\", "", content)

    return content

def parse_texts(files_path):
    # parse texts from folders
    section_texts = {}
    for path in Path(files_path).rglob('*.tex'):
        name = path.name.split(".")[0]
        with open(path, "r") as f:
            text = f.read()
            text = preprocess_latex(text)
        section_texts[name] = text
    return section_texts

def parse_section_names(files_path):
    section2name = {}

    with open(files_path, "r") as f:
        main_text = f.read()
    walker = LatexWalker(main_text)
    nodes, _, _ = walker.get_latex_nodes()

    for i, node in enumerate(nodes[2].nodelist):
        if isinstance(node, LatexMacroNode) and node.macroname == "section":
            section_name = node.nodeargd.argnlist[2].nodelist[0].chars
            
            try:
                include_node = nodes[2].nodelist[i + 2]
                if isinstance(include_node, LatexMacroNode) and include_node.macroname == "input":
                    section_number = include_node.nodeargd.argnlist[0].nodelist[0].chars.split("/")[-1].split(".")[0]
                    section2name[section_number] = section_name
                elif isinstance(include_node, LatexMacroNode) and include_node.macroname == "includegraphics":
                    continue
            except:
                continue
    return section2name

def parse_teormin_text(teormin_file):
    with open(teormin_file, "r") as f:
        teormin_text = f.read()

    teormin_name_text = []

    section_pattern = r"\\subsection\{(.+?)\}"
    sections = list(re.finditer(section_pattern, teormin_text))

    for s1, s2 in zip(sections, sections[1:]):
        s_name = s1.group(1)
        text_start = s1.end()
        text_end = s2.start()
        s_text = teormin_text[text_start: text_end]
        s_text = preprocess_latex(s_text)
        df_row = [s_name, s_text]
        teormin_name_text.append(df_row)
    return teormin_name_text


def parse_data(path_to_files = "./5_sem_ml", savepath = "data_for_ml.csv"):

    dataset_path = Path(path_to_files)
    
    texts_folder = dataset_path/"section/"
    section_texts = parse_texts(texts_folder) #for mapping section number to section text

    main_file_path = dataset_path/"main.tex"
    section2name = parse_section_names(main_file_path) # for mapping section number to section name

    # prepare data for dataframe ["section_name", "section_text"]

    teormin_file = dataset_path/"section/polidobro/teormin.tex"

    df_rows = parse_teormin_text(teormin_file)

    for s_num, s_name in section2name.items():
        s_text = section_texts[s_num]
        inp = [s_name, s_text]
        df_rows.append(inp)

    # make and save csv for model
    df = pd.DataFrame(df_rows, columns=["section_name", "section_text"])
    df.to_csv(savepath, header=False, index = False)