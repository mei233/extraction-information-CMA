Le script NB_model.py

Ce script a pour l'objectif d‘entraîner un model de classification sur les tweets de réchauffement climatique à partir de la méthode multinomial NB et la représentation de Tf-Idf. Et il peut également lire un model déjà enregistré et prédire la polarité des phrases. 
	1. OS : Windows, Mac, Linux
	2. Version du python : python3 ou plus 
	3. Librairies nécessaires: sklearn, pandas, nltk, matplotlib,joblib
	4. Mettre les fichiers de train et test dans le même dossier du script
 		Fichier de train : 
		Fichier de test : 
	5. Fichier(s) de model : NB_model.m , tfidfvectorizer_model.m . Ils doivent être dans le même dossier du script.
	6. Usage : 
	train_model = classifier_pipeline()
	Cela permet de créer une instance de la classe classifier_pipeline().

	(1) Pour entraîner un model, vous allez dans le script et au fond, et utilisez/decommenter le code au-dessous : 
    	train_model.pipeline(save_model=False)  

	'sava_model = False' veut dire de ne pas enregistrer le model dans vos ordinateur. 
	
	(2) Pour load a model entregistré, veuillez utiliser/decommenter le code au-dessous :
		train_model.load_model('NB_model.m','tfidfvectorizer_model.m')
    
    (3) Pour predire la polarité des phrases, veuillez utiliser/decommenter les codes au-dessous : 
		test_sentence = ["I am sure that the temperature is increasing, you don't notice that? ",
                   'Globle warming is not true']
		predict_label_list = train_model.predict_sentence(test_sentence)
   		print(predict_label_list)

   	7. Si vous voulez modifiation/changer les fichiers et train,  veuillez allez dans : 
   		 def __init__(self):
        	self.train_file = "train_file.csv"
        	self.test_file = "test_file.csv"

       De même, si vous voulez changer les fichiers de model, veuillez allez dans : 
       	def load_model(self,model_file,vector_file):
       		print(f"Loading model {model_file}...")
        	self.vectorizer = joblib.load(vector_file)
        	print(f"Loading model {vector_file}...")
        	self.Naive = joblib.load(model_file) 


	