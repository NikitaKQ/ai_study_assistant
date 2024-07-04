"""
Parsing .tex files into the database
"""
import re
import pandas as pd


class Parser:
    def __init__(self, path):
        self.path = path
        with open(path, 'r') as file:
            self.text = file.read()

    def parse(self, limiters):
        pattern = re.compile(r"\\" + limiters + r"\{([\s\S]*?)\}([\s\S]*?)(?=\\" + limiters + r")", re.DOTALL)
        matches = pattern.findall(self.text)
        res = []
        for match in matches:
            name = match[0].strip()
            data = match[1].strip()
            res.append([name, data])
        return res


if __name__ == "__main__":
    limiters = {
        "mathematical-analysis-colloquium-1.tex": "subsubsection",
        "mathematical-analysis-colloquium-2.tex": "defitem",
        "mathematical-analysis-colloquium-3.tex": "subsection",
        "mathematical-analysis-colloquium-4.tex": "subsection",
        "mathematical-analysis-exam-1.tex": "proofitem"
    }
    ans = pd.DataFrame(columns=["name", "data"])
    for j in ["mathematical-analysis-colloquium-1.tex", "mathematical-analysis-colloquium-2.tex",
              "mathematical-analysis-colloquium-3.tex", "mathematical-analysis-colloquium-4.tex",
              "mathematical-analysis-exam-1.tex"]:
        p = Parser(j)
        res = pd.DataFrame(p.parse(limiters[j]), columns=["name", "data"])
        ans = pd.concat([ans, res], ignore_index=True)
    ans.to_csv("db.csv")
    print(ans)
