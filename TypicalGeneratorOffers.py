# -*- coding: utf-8 -*-
"""
Author: weiyx15

Find typical generator offer curve from energy_market_offers2017.csv
input file:  energy_market_offers2017.csv
output file: generator_offers2017.csv
"""

import csv

csv_reader = csv.reader(open('GenerationOffers\energy_market_offers2017.csv','r'))
csv_writer = csv.writer(open('GenerationOffers\generator_offers2017.csv','w',\
                             newline=''), dialect='excel')
gen_name_set = set()
for row in csv_reader:
    gen_name = row[2]
    if gen_name not in gen_name_set: 
        data_piece = []
        for i in range(2,24):
            data_piece.append(row[i])
        csv_writer.writerow(data_piece)
        gen_name_set.add(gen_name)
print(len(gen_name_set))