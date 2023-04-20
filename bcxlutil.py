from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter, column_index_from_string

def xlsform(writer, sheet, cells = ["A", 1, "U", 15], fill = None, font = None, num_format = None, align = None):
    ws = writer.sheets[sheet]
    col_begin = column_index_from_string(cells[0])
    row_begin = cells[1] 
    col_end = column_index_from_string(cells[2]) +1
    row_end = cells[3] +1 
    for c in range(col_begin, col_end):
        for z in range(row_begin, row_end):
            if fill != None:
                ws.cell(row=z, column=c).fill = fill
            if font != None:
                ws.cell(row=z, column=c).font = font
            if num_format != None:
                ws.cell(row=z, column=c).number_format = num_format
            if align != None:
                ws.cell(row=z, column=c).alignment =  Alignment(horizontal=align)
                
