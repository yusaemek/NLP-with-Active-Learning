from AL_wlib import *

from AL_wlib import *
import pandas as pd
import numpy as np
plt_show = False

from warnings import filterwarnings
from sklearn.model_selection import train_test_split

def setup(s=1.0, nrows=1, ncols=1):
	import matplotlib
	matplotlib.rcParams.update({'figure.dpi': 1080})
	matplotlib.rcParams.update({'font.size': 15})

	matplotlib.rcParams.update({'legend.fontsize': 13})
	import __main__ as pc
	pc.default_figsize=(20, 10)
	pc.default_figsize = (pc.default_figsize[0]*s, pc.default_figsize[1]*s)
	import matplotlib.pyplot as plt
	plt.close()
	return plt.subplots(nrows, ncols, figsize=pc.default_figsize)
class PlotStyles:
	def __init__(self):
		self.markers = []
		self.markers += ['.']
		self.markers += ['1', '2', '3', '4', '+', 'x']
		self.markers += [4, 5, 6, 7, 8, 9, 10, 11]
		self.linestyles = [
	(0, (1, 1)),
	(0, (5, 10)),
	(0, (5, 5)),
	(0, (5, 1)),
	(0, (3, 10, 1, 10)),
	(0, (3, 5, 1, 5)),
	(0, (3, 1, 1, 1)),
	(0, (3, 5, 1, 5, 1, 5)),
	(0, (3, 10, 1, 10, 1, 10)),
	(0, (3, 1, 1, 1, 1, 1))
	]
		self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan','tab:black']
		self.reset()
	def reset(self):
		self.c_colors = 0
		self.c_markers = 0
		self.c_linestyles = 0
	def set_color_counter(self, c):
		self.c_colors = c
	def set_marker_counter(self, c):
		self.c_markers = c
	def set_linestyle_counter(self, c):
		self.c_linestyles = c
	def getNextStyle(self):
		retval = {}
		retval['color'] = self.colors[self.c_colors % len(self.colors)]
		retval['marker'] = self.markers[self.c_markers % len(self.markers)]
		retval['linestyle'] = self.linestyles[self.c_linestyles % len(self.linestyles)]
		self.c_colors += 1
		self.c_markers += 1
		self.c_linestyles += 1
		return retval


data_cumh = pd.read_pickle("./cumh_prep_econ.pkl")



X_train, X_test, y_train, y_test = train_test_split(data_cumh['text'], data_cumh['econ'],test_size =0.1,  random_state=0)

X_train=X_train.to_numpy()
X_test=X_test.to_numpy()
y_train=y_train.to_numpy()
y_test=	y_test.to_numpy()

initdata=10
query_size=25



selection=np.random.choice(X_train.shape[0], initdata, replace=False)
configuration=config(query_size, OneVsRestClassifier_logreg, infodensity, TfidfVectorizer,410, [], [5,(1,1)])
model=ALmodel(configuration,X_train,y_train,X_test,y_test,selection)

model.run()


configuration_r=config(query_size, OneVsRestClassifier_logreg, RandomSelection, TfidfVectorizer,410, [], [5,(1,1)] )
model_r=ALmodel(configuration_r,X_train,y_train,X_test,y_test,selection)

model_r.run()

configuration2=config(query_size, OneVsRestClassifier_logreg, MarginSelection, TfidfVectorizer,410, [], [5,(1,1)] )
model2=ALmodel(configuration2,X_train,y_train,X_test,y_test,selection)

model2.run()

configuration3=config(query_size, OneVsRestClassifier_logreg, graph, TfidfVectorizer,410, [], [5,(1,1)] )
model3=ALmodel(configuration3,X_train,y_train,X_test,y_test,selection)

model3.run()
print("Hey")


#configuration4=config(query_size, OneVsRestClassifier_logreg, quire, TfidfVectorizer,410, [], [5,(1,1)] )
#model4=ALmodel(configuration4,X_train,y_train,X_test,y_test,selection)
#model4.run()

ps = PlotStyles()



fig, ax = setup()


training_size=[m*query_size for m in range (len(model.accuracy_test))]

print( model.accuracy_test)
print( model_r.accuracy_test)
print( model2.accuracy_test)
print( model3.accuracy_test)
#print( model4.accuracy_test)

ax.plot(training_size, model_r.accuracy_test,**ps.getNextStyle(), label='random selection 25')
ax.plot(training_size, model.accuracy_test,**ps.getNextStyle(), label='infodensity selection 25')
ax.plot(training_size, model2.accuracy_test,**ps.getNextStyle(), label='margin selection 25')
ax.plot(training_size, model3.accuracy_test,**ps.getNextStyle(), label='graph selection 25')
#ax.plot(training_size, model4.accuracy_test,**ps.getNextStyle(), label='quire selection 25')




print("   (((0) ")


tit='Binary svm_svc, 4 class econ-tech, 25, frac=0.5'
ax.set(xlabel='queried size', ylabel='accuracy with test data',title=tit)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
ax.legend()
fig.savefig('logreg_4class.svg', format='svg', dpi=1080)