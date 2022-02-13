import pandas as pd
###### helper functions #######
def get_disease(disease):
	return DN[DN.disease == disease]["disease"].values[0]

def get_disease_name(disease_id):
	return DN[DN.disease_id == disease_id]["disease"].values[0]

def get_disease_id(disease):
	return DN[DN.disease == disease]["disease_id"].values[0]

def get_disease_ie(disease):
	return DN[DN.disease == disease]["ineficient_nutritions"].values[0]


####### dataset #######
FN = pd.read_csv("dataset/food_nutritions.csv")
DN = pd.read_csv("dataset/disease_nutrition.csv",encoding='unicode_escape')

####### Part-1 #######
users_disease = input("Enter Disease Name: ")
disease_name = get_disease(users_disease)
disease_id = get_disease_id(disease_name)
i= disease_id-101
p = DN.Precaution_1[i],DN.Precaution_2[i],DN.Precaution_3[i]
#print(DN.Medicene_1[i],DN.Medicene_2[i],DN.Medicene_3[i])
#print(DN.Precaution_1[i],DN.Precaution_2[i],DN.Precaution_3[i])
print(p)
#print(DN.iloc[i])         
disease_ie = get_disease_ie(disease_name)
#Convert disease-nutritions column value into list
dis_list = list(disease_ie.split(" "))
#print(dis_list)
#Convert food column into list
col_list = FN.columns.values.tolist()
#print(col_list)

      







