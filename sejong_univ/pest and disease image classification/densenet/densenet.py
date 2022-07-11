from dense_images import create_train_data, create_train_label
from dense_images import create_valid_data, create_valid_label
from dense_images import create_test_data, create_test_label

from dense_images import create_anth_data, create_anth_label
from dense_images import create_bact_data, create_bact_label
from dense_images import create_cmv_data, create_cmv_label
from dense_images import create_gray_data, create_gray_label
from dense_images import create_tswv_data, create_tswv_label
from dense_images import create_norm_data, create_norm_label
from dense_funcs import *

# Load image, label
train_img = create_train_data()
train_lbl = create_train_label()

valid_img = create_valid_data()
valid_lbl = create_valid_label()

test_img = create_test_data()
test_lbl = create_test_label()

anth_img = create_anth_data()
anth_lbl = create_anth_label()

bact_img = create_bact_data()
bact_lbl = create_bact_label()

cmv_img = create_cmv_data()
cmv_lbl = create_cmv_label()

gray_img = create_gray_data()
gray_lbl = create_gray_label()

tswv_img = create_tswv_data()
tswv_lbl = create_tswv_label()

norm_img = create_norm_data()
norm_lbl = create_norm_label()

#===o===o===o===o===o===o===o===o===o===o===#

train_img = train_img.astype(np.float64)
valid_img = valid_img.astype(np.float64)
test_img = test_img.astype(np.float64)

anth_img = anth_img.astype(np.float64)
bact_img = bact_img.astype(np.float64)
cmv_img = cmv_img.astype(np.float64)
gray_img = gray_img.astype(np.float64)
tswv_img = tswv_img.astype(np.float64)
norm_img = norm_img.astype(np.float64)

train_img = train_img / 255.0
valid_img = valid_img / 255.0
test_img = test_img / 255.0

anth_img = anth_img / 255.0
bact_img = bact_img / 255.0
cmv_img = cmv_img / 255.0
gray_img = gray_img / 255.0
tswv_img = tswv_img / 255.0
norm_img = norm_img / 255.0

#===o===o===o===o===o===o===o===o===o===o===#

img_input = Input(shape = (32, 32, 3))
x = densenet(4, img_input = img_input, n_filter = 16, bottleneck = False, dropout_rate = 0.2, weight_decay = 1e-4)
model = Model(inputs = img_input, outputs = x)

# DenseNet Model Architecture
#os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#plot_model(model, to_file = 'C:/Users/lee/Desktop/grad_paper7/dense_archi.jpeg', show_shapes = True)

opt_adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
model.compile(loss = 'categorical_crossentropy', optimizer = opt_adam, metrics = ['accuracy'])

s_time = datetime.datetime.now()
history = model.fit(train_img, train_lbl, batch_size = 32, epochs = 150, shuffle = True, validation_data = (valid_img, valid_lbl))
e_time = datetime.datetime.now()
print('Model learn time :', e_time - s_time)
print('')

#===o===o===o===o===o===o===o===o===o===o===#

scores = model.evaluate(test_img, test_lbl)
print('')
print('Accuracy of DenseNet model using test image data')
print("%s : %.4f%%" % (model.metrics_names[0], scores[0])) # score
print("%s : %.2f%%" % (model.metrics_names[1], scores[1] * 100)) # accuracy

#===o===o===o===o===o===o===o===o===o===o===#

# Predict Densenet model
pred_test = model.predict(test_img)
true_test = np.argmax(test_lbl[:], axis = 1)
pred_test = np.argmax(pred_test[:], axis = 1)

print('')
print('Test data Classification report')
cls_rpt = classification_report(true_test, pred_test)
print(cls_rpt)

print('Test data Confusion matrix')
cfs_mt = confusion_matrix(true_test, pred_test)
print(cfs_mt)

#===o===o===o===o===o===o===o===o===o===o===#

pred_anth = model.predict(anth_img)
true_anth = np.argmax(anth_lbl[:], axis = 1)
pred_anth = np.argmax(pred_anth[:], axis = 1)

print('')
print('Anthracnose data Classification report')
cls_rpt_anth= classification_report(true_anth, pred_anth)
print(cls_rpt_anth)

print('Anthracnose data Accuracy')
acc_anth = accuracy_score(true_anth, pred_anth)
print("%.2f%%" % (acc_anth * 100))

print('')
print('Anthracnose data Confusion matrix')
cfs_mt_anth = confusion_matrix(true_anth, pred_anth)
print(cfs_mt_anth)

#===o===o===o===o===o===o===o===o===o===o===#

pred_bact = model.predict(bact_img)
true_bact = np.argmax(bact_lbl[:], axis = 1)
pred_bact = np.argmax(pred_bact[:], axis = 1)

print('')
print('Bacterialspot data Classification report')
cls_rpt_bact= classification_report(true_bact, pred_bact)
print(cls_rpt_bact)

print('Bacterialspot data Accuracy')
acc_bact = accuracy_score(true_bact, pred_bact)
print("%.2f%%" % (acc_bact * 100))

print('')
print('Bacterialspot data Confusion matrix')
cfs_mt_bact= confusion_matrix(true_bact, pred_bact)
print(cfs_mt_bact)

#===o===o===o===o===o===o===o===o===o===o===#

pred_cmv = model.predict(cmv_img)
true_cmv = np.argmax(cmv_lbl[:], axis = 1)
pred_cmv = np.argmax(pred_cmv[:], axis = 1)

print('')
print('CMV data Classification report')
cls_rpt_cmv = classification_report(true_cmv, pred_cmv)
print(cls_rpt_cmv)

print('CMV data Accuracy')
acc_cmv = accuracy_score(true_cmv, pred_cmv)
print("%.2f%%" % (acc_cmv * 100))

print('')
print('CMV data Confusion matrix')
cfs_mt_cmv= confusion_matrix(true_cmv, pred_cmv)
print(cfs_mt_cmv)

#===o===o===o===o===o===o===o===o===o===o===#

pred_gray = model.predict(gray_img)
true_gray = np.argmax(gray_lbl[:], axis = 1)
pred_gray = np.argmax(pred_gray[:], axis = 1)

print('')
print('Graymold data Classification report')
cls_rpt_gray = classification_report(true_gray, pred_gray)
print(cls_rpt_gray)

print('Graymold data Accuracy')
acc_gray = accuracy_score(true_gray, pred_gray)
print("%.2f%%" % (acc_gray * 100))

print('')
print('Graymold data Confusion matrix')
cfs_mt_gray = confusion_matrix(true_gray, pred_gray)
print(cfs_mt_gray)

#===o===o===o===o===o===o===o===o===o===o===#

pred_tswv = model.predict(tswv_img) # test data에 대한 prediction 값
true_tswv = np.argmax(tswv_lbl[:], axis = 1)
pred_tswv = np.argmax(pred_tswv[:], axis = 1)

print('')
print('TSWV data Classification report')
cls_rpt_tswv = classification_report(true_tswv, pred_tswv)
print(cls_rpt_tswv)

print('TSWV data Accuracy')
acc_tswv= accuracy_score(true_tswv, pred_tswv)
print("%.2f%%" % (acc_tswv * 100))

print('')
print('TSWV data Confusion matrix')
cfs_mt_tswv = confusion_matrix(true_tswv, pred_tswv)
print(cfs_mt_tswv)

#===o===o===o===o===o===o===o===o===o===o===#

pred_norm = model.predict(norm_img)
true_norm = np.argmax(norm_lbl[:], axis = 1)
pred_norm = np.argmax(pred_norm[:], axis = 1)

print('')
print('Normal data Classification report')
cls_rpt_norm = classification_report(true_norm, pred_norm)
print(cls_rpt_norm)

print('Normal data Accuracy')
acc_norm = accuracy_score(true_norm, pred_norm)
print("%.2f%%" % (acc_norm * 100))

print('')
print('Normal data Confusion matrix')
cfs_mt_norm = confusion_matrix(true_norm, pred_norm)
print(cfs_mt_norm)

#===o===o===o===o===o===o===o===o===o===o===#

# Summarize Fit DenseNet Model for Accuracy and Loss
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

#plt.title('')
ta = ax1.plot(history.history['acc'], 'r-', label = 'train_acc')
va = ax1.plot(history.history['val_acc'], 'b-', label = 'valid_acc')

tl = ax2.plot(history.history['loss'], 'y-', label = 'train_loss')
vl = ax2.plot(history.history['val_loss'], 'g-', label = 'valid_loss')

lines = ta + va + tl + vl
legends = [l.get_label() for l in lines]
ax1.legend(lines, legends, loc = 0)

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
plt.show()
