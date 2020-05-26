
from trumpDataset.basisFuncs import *

def populateDeletedColumn():
	mycursor.execute("select id from "+mainTable)
	updateFormula = "UPDATE "+mainTable+" SET deleted = %s WHERE id = %s";

	ids = [int(id[0]) for id in mycursor.fetchall()]
	tuples = [];
	print("iterations: ",int(len(ids)/100)+1)
	for i in range(int(len(ids)/100)+1):
		idPartition = ids[i*100:(i+1)*100];
		# idPartition = [ids[0],869766994899468288,ids[1]]
		responses = getTweetsById(idPartition);
		idIntsFromTwitter = [int(resp.id_str) for resp in responses]
		deletedInts = [int(id in idIntsFromTwitter) for id in idPartition];
		tuples.extend([(deletedInts[j],idPartition[j]) for j in range(len(deletedInts))])
		print("finished iteration ",i);
		print(list(zip(responses,deletedInts)))

	mycursor.executemany(updateFormula,tuples);
	print("executed")
	mydb.commit()
