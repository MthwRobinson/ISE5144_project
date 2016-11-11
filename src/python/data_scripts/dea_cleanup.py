import openpyxl as pyxl
import csv

def main():
    folder = '/home/matt/ISE5144_project/output/dea/'
    dea = 'dea_results.xlsx'
    wb = pyxl.load_workbook(folder+dea, read_only = True,
            data_only = True)
    worksheets = wb.worksheets
    for worksheet in worksheets:
        title = worksheet.title
        csv_file = title.lower().replace(' ','_') + '.csv'
        csv_file = folder + csv_file
        if csv_file != 'lambda.csv':
            print('Writing: ', csv_file)
            with open(csv_file, 'wb') as f:
                c = csv.writer(f)
                for r in worksheet.rows:
                    c.writerow([cell.value for cell in r])

if __name__ == '__main__':
    main()
