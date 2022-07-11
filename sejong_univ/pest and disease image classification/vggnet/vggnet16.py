from vgg_images import create_train_data, create_train_label
from vgg_images import create_valid_data, create_valid_label
from vgg_images import create_test_data, create_test_label

from vgg_images import create_fire_data, create_fire_label
from vgg_images import create_scab_data, create_scab_label
from vgg_images import create_blac_data, create_blac_label
from vgg_images import create_norm_data, create_norm_label
from vgg16_funcs import *

# Load image, label
train_img = create_train_data()
train_lbl = create_train_label()

valid_img = create_valid_data()
valid_lbl = create_valid_label()

test_img = create_test_data()
test_lbl = create_test_label()

fire_img = create_fire_data()
fire_lbl = create_fire_label()

scab_img = create_scab_data()
scab_lbl = create_scab_label()

blac_img = create_blac_data()
blac_lbl = create_blac_label()

norm_img = create_norm_data()
norm_lbl = create_norm_label()

#===o===o===o===o===o===o===o===o===o===o===#

train_img = train_img.astype(np.float64)
valid_img = valid_img.astype(np.float64)
test_img = test_img.astype(np.float64)

fire_img = fire_img.astype(np.float64)
scab_img = scab_img.astype(np.float64)
blac_img = blac_img.astype(np.float64)
norm_img = norm_img.astype(np.float64)

train_img = train_img / 255.0
valid_img = valid_img / 255.0
test_img = test_img / 255.0

fire_img = fire_img / 255.0
scab_img = scab_img / 255.0
blac_img = blac_img / 255.0
norm_img = norm_img / 255.0

#===o===o===o===o===o===o===o===o===o===o===#

# VGG16 Model
#input_shape = train_img.shape[1:] # input_shape = (32, 32, 3)
img_input = Input(shape = (224, 224, 3)) # 224x224 RGB image
x = VGG16(n_classes = 3, img_input = img_input, dropout_rate = 0.5)
model = Model(img_input, x)

#===o===o===o===o===o===o===o===o===o===o===#

opt_adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
model.compile(loss = 'categorical_crossentropy', optimizer = opt_adam, metrics = ['accuracy'])
#vgg_model.compile(loss = 'categorical_crossentropy', optimizer = opt_adam, metrics = ['accuracy'])

s_time = datetime.datetime.now() # training start time
history = model.fit(train_img, train_lbl, batch_size = 32, epochs = 2, shuffle = True, validation_data = (valid_img, valid_lbl))
e_time = datetime.datetime.now() # training end time
print('Model learn time : ', e_time - s_time)
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

pred_fire = model.predict(fire_img)
true_fire = np.argmax(fire_lbl[:], axis = 1)
pred_fire = np.argmax(pred_fire[:], axis = 1)

print('')
print('Fireblight data Classification report')
cls_rpt_fire = classification_report(true_fire, pred_fire)
print(cls_rpt_fire)

print('Fireblight data Accuracy')
acc_fire = accuracy_score(true_fire, pred_fire)
print("%.2f%%" % (acc_fire * 100))

print('')
print('Fireblight data Confusion matrix')
cfs_mt_fire = confusion_matrix(true_fire, pred_fire)
print(cfs_mt_fire)

#===o===o===o===o===o===o===o===o===o===o===#

pred_scab = model.predict(scab_img) # test data에 대한 prediction 값
true_scab = np.argmax(scab_lbl[:], axis = 1)
pred_scab = np.argmax(pred_scab[:], axis = 1)

print('')
print('Scab data Classification report')
cls_rpt_scab = classification_report(true_scab, pred_scab)
print(cls_rpt_scab)

print('Scab data Accuracy')
acc_scab = accuracy_score(true_scab, pred_scab)
print("%.2f%%" % (acc_scab * 100))

print('')
print('Scab data Confusion matrix')
cfs_mt_scab = confusion_matrix(true_scab, pred_scab)
print(cfs_mt_scab)

#===o===o===o===o===o===o===o===o===o===o===#

pred_blac = model.predict(blac_img) # test data에 대한 prediction 값
true_blac = np.argmax(blac_lbl[:], axis = 1)
pred_blac = np.argmax(pred_blac[:], axis = 1)

print('')
print('Blacknecroticleafspot data Classification report')
cls_rpt_blac = classification_report(true_blac, pred_blac)
print(cls_rpt_blac)

print('Blacknecroticleafspot data Accuracy')
acc_blac = accuracy_score(true_blac, pred_blac)
print("%.2f%%" % (acc_blac * 100))

print('')
print('Blacknecroticleafspot data Confusion matrix')
cfs_mt_blac = confusion_matrix(true_blac, pred_blac)
print(cfs_mt_blac)

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

plt.title('Accuracy and loss of pear train, validation data')
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