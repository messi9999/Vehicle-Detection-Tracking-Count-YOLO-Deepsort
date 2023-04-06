import os
import datetime
import openpyxl

class Save_data:
    def __init__(self):
        self.filename = './ExcelOutput/data4.xlsx'
        self.list_direction = []
        
    def save_data(self, filename, directions):
        self.list_direction = directions.copy()
        path = os.getcwd()
        isFile = os.path.isfile(path + '/' + filename)
        now = datetime.datetime.now()
        if isFile == False:
            wb = openpyxl.Workbook()
            ws = wb.active
            row = ["Direction", "Car", "Motorbike", "Bus", "Truck", "Time"]
            ws.append(row)
        else:
            wb = openpyxl.load_workbook(filename)
            ws = wb.active

        for i, direc in enumerate(self.list_direction):
            match i:
                case 0:
                    direc.insert(0, "Direction A -> B")
                case 1:
                    direc.insert(0, "Direction B -> A")
                case 2:
                    direc.insert(0, "Direction A -> C")
                case 3:
                    direc.insert(0, "Direction C -> A")
                case 4:
                    direc.insert(0, "Direction A -> D")
                case 5:
                    direc.insert(0, "Direction D -> A")
                case 6:
                    direc.insert(0, "Direction B -> C")
                case 7:
                    direc.insert(0, "Direction C -> B")
                case 8:
                    direc.insert(0, "Direction B -> D")
                case 9:
                    direc.insert(0, "Direction D -> B")
                case 10:
                    direc.insert(0, "Direction C -> D")
                case 11:
                    direc.insert(0, "Direction D -> C")
                
            direc.insert(5, now)
            ws.append(direc)
        ws.append(['~~~','~~~','~~~','~~~','~~~','~~~'])
        ws.append(['','','','','',''])

        wb.save(filename)
        return True


