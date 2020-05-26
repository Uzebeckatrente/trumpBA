
categories = ["criticism","encouragement/tribute","gaffes","vitriol"]
counts = [0]*len(categories);
letters = {"c":0,"e":1,"t":1,"g":2,"v":3}


inputLetterString = None;
while True:
	inputLetterString = str(input());
	if inputLetterString == "done":
		print(categories)
		print(counts)
		exit(1)
	elif len(inputLetterString) == 1:
		if inputLetterString in letters:
			counts[letters[inputLetterString]] += 1
			print("entered text for ",categories[letters[inputLetterString[0]]])
			continue
		else:
			print("you've entered a character that we don't know yet! try again")
			continue
	else:
		if inputLetterString in categories:
			print("already exists! try again");
			continue;
		elif inputLetterString[0] in letters:
			print("category ",categories[letters[inputLetterString[0]]]," already starts with letter",inputLetterString[0])

		else:

			categories.append(inputLetterString)
			letters[inputLetterString[0]]=len(categories)-1
			counts.append(1);

