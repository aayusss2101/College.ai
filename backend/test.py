import camelot

tables=camelot.read_pdf("time table.pdf")
print(len(tables))
df=tables[0].df
# print(df[0][0])

# Department
# print(df[1][0])

print(df[2][0])
# print(df[4][0])