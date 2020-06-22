##source for approval ratings: https://github.com/fivethirtyeight/data/tree/master/trump-approval-ratings

from .processTweetData import *

def sourceApprovalRatings():
	f = open("/Users/f002nb9/Downloads/approval_topline.csv");
	insertFormula = "INSERT INTO approvalRatings (date, approveEstimate, disapproveEstimate) VALUES (%s, %s, %s)"
	mycursor.execute("delete from approvalRatings")
	firstLine = f.readline()
	print(firstLine);
	tuples = [];

	for line in f:
		line = line.split(",");
		if line[1].lower() != "all polls":
			continue;


		upSegs = ["0" + seg if len(seg) == 1 else seg for seg in line[2].split("/")]
		date = upSegs[2]+"-"+"-".join(upSegs[:2])

		appr = float(line[3])
		disappr = float(line[6])
		dateMySQL = date+ " 00:00:01";
		tuples.append((dateMySQL,appr,disappr));
		# break;




	mycursor.executemany(insertFormula,tuples);
	mydb.commit()
	print("gone and executed")