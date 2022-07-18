
import os
pdf_list = os.listdir(r'C:\rnd_project\Cited_Author_M_Thelwall')
# print(list)
import pdfplumber
ext_txt = []
for i in range(len(pdf_list)):
    file_name = "C:\\rnd_project\\Cited_Author_M_Thelwall\\" + str(pdf_list[i])
    with pdfplumber.open(file_name) as pdf:
        first_page = pdf.pages[2]
        ext_txt.append(first_page.extract_text())
print(ext_txt[0])