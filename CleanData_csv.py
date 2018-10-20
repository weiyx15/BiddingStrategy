# -*- coding: utf-8 -*-
"""
Author: weiyx15

Data cleaning for PJM generation offers
csv version

input files:  energy_market_offers(0) ~ energy_market_offers(6)
output files: energy_market_offers2017 & energy_market_offers2018
"""

import csv

out0 = open('GenerationOffers\energy_market_offers2017.csv', 'w', newline='')
out1 = open('GenerationOffers\energy_market_offers2018.csv', 'w', newline='')
csv_writer0 = csv.writer(out0, dialect='excel')
csv_writer1 = csv.writer(out1, dialect='excel')
# move all the items of year 2017 to 'energy_market_offers2017.csv'
# move all the items of year 2018 to 'energy_market_offers2018.csv'
for i in range(7):
    filename = 'GenerationOffers\energy_market_offers(' + str(i) + ').csv'
    csv_reader = csv.reader(open(filename, 'r'))
    data = []
    for row in csv_reader:
        s_datetime = row[0]
        s_date = s_datetime.split(' ')[0]
        if '/' in s_date:
            data.append(row)
    data.reverse()
    for rev_row in data:
        s_datetime = rev_row[0]
        s_date = s_datetime.split(' ')[0]
        s_year = s_date.split('/')[2]
        if s_year == '2017':
            csv_writer0.writerow(rev_row)
        elif s_year == '2018':
            csv_writer1.writerow(rev_row)
out0.close()
out1.close()
