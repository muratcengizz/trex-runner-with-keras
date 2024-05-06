import os
import glob
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TrexAI:
    def __init__(self, imgs, width, height, model_name):
        self.imgs = imgs
        self.width = width
        self.height = height
        self.model_name = model_name

    def preprocess(self):
        """
            Veri seti toplanirken her goruntu dosyasinin ismi sahip oldugu etiket ile isimlendirildi.
            Bu yuzden bu fonksiyon, dosya ismini etiket olarak kaydeder. Goruntu dosyalarini da 
            keras tarafindan beklenen formata donusturur.
        """
        X, y = [], []
        for img in self.imgs:
            filename = os.path.basename(p=img) # dosya ismini aldik.
            label = filename.split('_')[0] # dosya isminden etiketi kesip label isimli degiskene kaydettik.
            open_image = Image.open(fp=img) # bytes formatindaki goruntu dosyasini actik.
            convert_to_grayscale = open_image.convert(mode='L') # goruntuyu grayscale formata cevirdik. Bu sayede renkler gider ve sadece parlaklık kalir.
            resize_image = convert_to_grayscale.resize(size=(self.width, self.height)) # goruntuyu yeniden boyutlandirdik.
            bytes_to_array = np.array(object=resize_image) # bytes-np.array'i donusumu yaptik.
            X.append(bytes_to_array)
            y.append(label)
        return X, y

    def reshape_images(self, X):
        """
            Liste formatinda gelen X degiskeni, icerisinde goruntu dosylarini array formatinda barindiriyor.
            Bu fonksiyon, bu listeyi np.array'ine donusturur. Daha sonra bu array'i
            icerisinde 300 oge bulunduran, 125 satir ve 50 sütundan olusan 1 kanalli bir array'e donusturur.
            300 oge = Ben 300 adet veri topladim. X degiskeninin icerisinde 300 adet veri var.
            125 satir = width degeri
            50 sütun = height degeri
            1 kanal = Goruntu dosyalarinin gray scale yani gri tonlamali yapida oldugunu belirtiyor.
        """
        X = np.array(X)
        X = X.reshape(X.shape[0], self.width, self.height, 1)
        return X
    
    def onehot_labels(self, y):
        """
            etiketler: up, down, right
            Bu fonksiyon, y degiskeni icerisinde gelen etiketleri sayisal formata donusturur.
            mesela; up=0, down=1, right=2 
            daha sonra bu sayisal degerler one-hot formatina donusturulur. Ornegin,
            up = [1, 0, 0],    down=[0, 1, 0],     right=[0, 0, 1]
            şeklinde etiketlerimiz olusur. Bu etiketleme sistemi keras tarafindan kabul edilir.
        """
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y=y) # string etiket degerleri sayisal olarak deger alirlar. up=0, down=1, right=2 gibi.
        onehot_encoder = OneHotEncoder(sparse_output=False) # sparse_output parametresi True olarak gelir ve sayisal etiketleri csr_matrix'e donusturur. Fakat biz bunu istemiyoruz. Yukarida anlattigim gibi cikti almak istiyoruz. Bu yuzden False degerini veriyoruz.
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1) # tamsayi etiketleri sütun vektörlerine cevirir.
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded) # tamsayi etiketleri one-hot vektörlerine donusturur.
        return onehot_encoded
    
    def train_test_splitt(self, X, y, test_size, random_state):
        """
            Bu fonksiyon, X (goruntuler) ve y (etiketler) 'den olusan arrayleri test_size ven random_state
            degerlerini kullanarak ayirma islemi yapar.
        """
        train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return train_X, test_X, train_Y, test_Y
    
    """
        Suanda verilerimizi on isleyerek egitime hazirladik. Simdi ise evrisimli sinir agi
        mimarileri kurarak egitimler gerceklestirecegiz ve farklari inceleyecegiz.
    """
    
    def architecture1(self):
        model = Sequential() # Sinir agi modeli olusturduk.
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(self.width, self.height, 1))) # modele evrisimli sinir agi katmani ekledik. İlk katmanda modelin input_shape'ini belirtmeliyiz.
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # maksimum havuzlama katmani, giris verilerinin boyutunu azaltmak ve ozelliklerin ortusmesini azaltarak ozellik haritasinin boyutunu kucultmek icin kullanilir.
        model.add(Dropout(rate=0.25)) # ağın her bir egitim orneginde (batch), belirli bir olasilikla rastgele secilen bir kismini (nöron) de-aktive ederek ağı güclendirir. Bu, ağın aşırı uyum (overfitting) yapmasını onlemeye yardimci olur.
        model.add(Flatten()) # Görüntü verilerini düzleştirmek için kullanılır. 2D görüntü verilerini tam bağlı katmanlara (Dense) gelmeden hemen önce sinir ağına (Dense) girdi olarak verebilmek için tek boyutlu bir diziye dönüştürür.
        model.add(Dense(units=128, activation='relu')) # sinir ağı modelindeki gizli katmanları veya çıkış katmanını oluşturmak için kullanılır. Veri üzerinde öğrenme yapılan kısımdır.
        model.add(Dropout(rate=0.4))
        model.add(Dense(units=3, activation='softmax')) # çıkış katmanı. Tahmin işlemi bu katmanda yapılıyor.
        return model

    def architecture2(self):
        model = Sequential()
        #block 1
        model.add(Input(shape=(self.width, self.height, 1)))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        #block 2
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        #block 3
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten layer
        model.add(Flatten())

        # Hidden layers
        model.add(Dense(units=128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(units=3, activation='softmax'))
        return model

    def architecture3(self):
        model = Sequential()
        #block 1
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(self.width, self.height, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.2))

        #block 2
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.3))

        #block 3
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.4))

        # Flatten layer
        model.add(Flatten())

        # Hidden layers
        model.add(Dense(units=128, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=3, activation='softmax'))
        return model
    

  
    def train_model(self, model, train_X, train_Y):
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model = model.fit(train_X, train_Y, epochs=50, batch_size=64)
        return model
    
    def train_augmented_model(self, model, train_X, train_Y, test_X, test_Y):
        datagen = ImageDataGenerator(
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            horizontal_flip = True,
            rotation_range = 20,
            shear_range = 0.1
        )
        train_iteration = datagen.flow(train_X, train_Y, batch_size=64)
        #steps = int(train_X.shape[0] / 64)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(train_iteration, epochs=250, batch_size=64, validation_data=(test_X, test_Y))
        return history
    
    def evaluate_model(self, model, train_X, train_Y, test_X, test_Y):
        score_train = model.evaluate(train_X, train_Y)
        score_test = model.evaluate(test_X, test_Y)
        return score_train, score_test
    
    def print_weight_file(self, model):
        model.save(f'{self.model_name}_arch1.h5')
        open('model_new.json', 'w').write(model.to_json())
        model.save_weights('trex_weight_new.h5')


    def process(self):
        X, y = self.preprocess()
        X = self.reshape_images(X=X)
        print(X.shape, X[0].shape)
        y = self.onehot_labels(y=y)
        train_X, test_X, train_Y, test_Y = self.train_test_splitt(X=X, y=y, test_size=0.25, random_state=10)

        # model 1
        model1_arch = self.architecture1()
        history1 = self.train_model(model=model1_arch, train_X=train_X, train_Y=train_Y)
        train_score1, test_score1 = self.evaluate_model(model=model1_arch, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
        self.print_weight_file(model=model1_arch)
        # evalute fonksiyonundan 2 adet liste return ediliyor. [train_loss, train_acc], [test_loss, test_acc]
        
        # # model 2
        # model2_arch = self.architecture2()
        # history2 = self.train_model(model=model2_arch, train_X=train_X, train_Y=train_Y)
        # train_score2, test_score2 = self.evaluate_model(model=model2_arch, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
        # self.print_weight_file(model=model2_arch)
        # # model 3
        # model3_arch = self.architecture3()
        # history3 = self.train_model(model=model3_arch, train_X=train_X, train_Y=train_Y)
        # train_score3, test_score3 = self.evaluate_model(model=model3_arch, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)

        # # augmented model
        # model1_arch = self.architecture1()
        # model_aug = self.train_augmented_model(model=model1_arch, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
        # train_score_aug, test_score_aug = self.evaluate_model(model=model1_arch, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
        # print(train_score_aug, test_score_aug)
        # print(model1_arch.summary())
        # self.print_weight_file(model=model1_arch)
   
        # summary = {
        #     "Model 1": {'Train Accuracy Score': train_score1[1], 'Train Loss Score': train_score1[0], 'Test Accuracy Score': test_score1[1], 'Test Loss Score': test_score1[0], 'Model Summary': model1_arch.summary()},
        #     "Model 2": {'Train Accuracy Score': train_score2[1], 'Train Loss Score': train_score2[0], 'Test Accuracy Score': test_score2[1], 'Test Loss Score': test_score2[0], 'Model Summary': model2_arch.summary()},
        #     "Model 3": {'Train Accuracy Score': train_score3[1], 'Train Loss Score': train_score3[0], 'Test Accuracy Score': test_score3[1], 'Test Loss Score': test_score3[0], 'Model Summary': model3_arch.summary()},
        #     "Augmented Model": {'Train Accuracy Score': train_score_aug[1], 'Train Loss Score': train_score_aug[0], 'Test Accuracy Score': test_score_aug[1], 'Test Loss Score': test_score_aug[0]}
        # }

         
        # score_list = [test_score1[1], test_score2[1], test_score3[1]]
        # if max(score_list) == test_score1[1]:
        #     self.print_weight_file(model=model1_arch)
        #     print(f'Model 1 en yüksek skora sahip. Model 1 kaydedildi.\nEğitim skoru: {train_score1[1]*100:.2f}\nTest skoru: {test_score1[1]*100:.2f}')
        # elif max(score_list) == test_score2[1]:
        #     self.print_weight_file(model=model2_arch)
        #     print(f'Model 2 en yüksek skora sahip. Model 2 kaydedildi.\nEğitim skoru: {train_score2[1]*100:.2f}\nTest skoru: {test_score2[1]*100:.2f}')
        # elif max(score_list) == test_score3[1]:
        #     self.print_weight_file(model=model3_arch)
        #     print(f'Model 3 en yüksek skora sahip. Model 3 kaydedildi.\nEğitim skoru: {train_score3[1]*100:.2f}\nTest skoru: {test_score3[1]*100:.2f}')
        # print(summary)


imgs = glob.glob(pathname='./images/*.png')
width = 250
height = 100
p1 = TrexAI(imgs=imgs, width=width, height=height, model_name='trexai')
p1.process()
