import csv 
# change filname for each dataset being merged
filename = "SYB62_1_201907_Population, Surface Area and Density.csv"
# filename = "SYB62_245_201905_Public Expenditure on Education.csv"
# filename = "SYB62_309_201906_Education.csv"
# filename = "test.csv"


fields = [] 
rows = [] 

with open(filename, 'r') as csvfile: 
	csvreader = csv.reader(csvfile) 
	
	# extracting field names through first row 
	fields = csvreader.next() 

	# extracting each data row one by one 
	for row in csvreader: 
		rows.append(row) 

# change 3rd param with selected param from dataset
col_headings = ['Region/Country/Area', 'Year', "Population mid-year estimates (millions)"]

print('' + ','.join(c for c in col_headings)) 


# change csv file to write to
with open('population.csv', mode='w') as output:
    output_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(col_headings)
    for row in rows: 
		if row[3] in col_headings:
			output_writer.writerow([row[1], row[2], row[4]])
