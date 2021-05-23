import csv

log_dir = r"F:\code\pythoncode\test_acc_data\mini\SA\a03_data_300\train.log"
csv_name = 'loss2.csv'


with open(log_dir,'r') as f:
    lines = f.readlines()
    f.close()

with open(csv_name, 'w', encoding='utf-8', newline='') as f:
    csvwriter = csv.writer(f, dialect='excel')
    csvwriter.writerow(["epoch", "Detector_loss", "D_loss", "encoder2_loss", "lr"])
    for i in lines:
        if 'Detector_loss' in i and 'test' not in i:
            i = i.split(' ')
            epoch = i[3][6:]
            Detector_loss = i[4][13:]
            D_loss = i[5][7:]
            encoder2_loss = i[6][7:]
            lr = i[7][3:]
            csvwriter.writerow([epoch, Detector_loss, D_loss, encoder2_loss, lr])
    f.close()
