import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# data mdd
df1 = pd.read_excel('/path_to/REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx', sheet_name='MDD')
# data controls
df2 = pd.read_excel('/path_to/REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx', sheet_name='Controls')
df = pd.concat([df1, df2])

# age groups
bins = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# create pdf
with PdfPages('plots.pdf') as pdf:
    for i in [df1, df2]:
        age_distribution = pd.cut(i['Age'], bins=bins).value_counts().sort_index()

        fig, ax = plt.subplots()
        rects = ax.bar(age_distribution.index.astype(str), age_distribution.values, color='r')
        plt.xlabel('Age')
        plt.ylabel('Counter')
        if i == df1:
            plt.title('distribution: age-MDD')
        else
            plt.title('distribution: age-Controls')
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        pdf.savefig(fig)
        plt.close(fig)