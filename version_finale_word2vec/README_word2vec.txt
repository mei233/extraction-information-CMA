Le script LSTM_word2Vec.py

Ce script a pour l'objectif d'entraîner un model de classification sur les tweets de réchauffement climatique à partir de la représentation de Word2vec et la méthode LSTM. Et il peut également lire un model déjà enregistré et prédire la polarité des phrases. 
	1. OS : Windows, Mac, Linux
	2. Version du python : python3 ou plus 
	3. Librairies nécessaires: sklearn, gensim, keras, pandas, numpy, nltk, matplotlib, TenserFlow
	4. Mettre les fichiers de train et test dans le même dossier du script
 		Fichier de train : 
		Fichier de test : 
	5. Fichier(s) de model : lstm_w2v_model.h5 , word2vec_twitter_tokens.bin (télécharger ce pre-trained model par ici : https://drive.google.com/file/d/1lw5Hr6Xw0G0bMT1ZllrtMqEgCTrM7dzc/view)
	les fichiers de model doivent être mis dans le même dossier du script 

	6. Usage : 
	lstm_w2v = LstmWord2vec()
	Cela permet de créer une instance de la classe LstmWord2vec().

	(1) Pour entraîner un model, vous allez dans le script et au fond, et utilisez/decommenter le code au-dessous : 
    	lstm_w2v.train_lstm(save_model=False,plot=False) 

	'sava_model = False' veut dire de ne pas enregistrer le model dans vos ordinateur. 
	
	(2) Pour load a model entregistré, veuillez utiliser/decommenter le code au-dessous :
		lstm_w2v.load_model('lstm_w2v_model.h5')
    
    (3) Pour predire la polarité des phrases, veuillez utiliser/decommenter les codes au-dessous : 
		test_sentence = ["I am sure that the temperature is increasing, you don't notice that? ",
                   'Globle warming is not true']
		predict_label_list = lstm_w2v.predict_sentence(test_sentence)
    	print(predict_label_list)

    7. Si vous voulez modifiation/changer les fichiers et train, ou pre-trained word2vec model,  veuillez allez dans : 
   		 def __init__(self):
        	self.train_file = "train_file_word2vec.csv"
        	self.test_file = "test_file_word2vec.csv"
        	self.w2v_model_file = "word2vec_twitter_tokens.bin"

       De même, si vous voulez changer le fichier de model, veuillez allez dans : 
       	 def load_model(self,load_model_file):
        	print(f"Loading model {load_model_file}...")
        	self.lstm_model = load_model(load_model_file)

