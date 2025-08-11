# Modul ini bertugas untuk memuat data dari berbagai sumber dan menyiapkannya untuk pemrosesan lebih lanjut.
# Cocok digunakan untuk proyek Machine Learning, Deep Learning, atau AI.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, data_path, data_type="csv"):
        """
        Inisialisasi DataLoader
        :param data_path: Path ke file atau folder dataset
        :param data_type: Jenis data ("csv", "excel", "image")
        """
        self.data_path = data_path
        self.data_type = data_type.lower()
        self.data = None

    def load_data(self):
        """
        Memuat data sesuai dengan tipe yang dipilih
        """
        if self.data_type == "csv":
            self.data = pd.read_csv(self.data_path)
        elif self.data_type == "excel":
            self.data = pd.read_excel(self.data_path)
        elif self.data_type == "image":
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Folder {self.data_path} tidak ditemukan.")
        else:
            raise ValueError("Jenis data tidak dikenali. Gunakan 'csv', 'excel', atau 'image'.")
        return self.data

    def split_data(self, test_size=0.2, random_state=42):
        """
        Membagi dataset menjadi data latih dan data uji
        Hanya berlaku untuk data tabular (csv/excel)
        """
        if self.data_type in ["csv", "excel"]:
            train, test = train_test_split(self.data, test_size=test_size, random_state=random_state)
            return train, test
        else:
            raise TypeError("Split data hanya berlaku untuk data tabular.")

    def normalize_data(self, feature_columns):
        """
        Normalisasi fitur numerik menggunakan MinMaxScaler
        """
        if self.data_type in ["csv", "excel"]:
            scaler = MinMaxScaler()
            self.data[feature_columns] = scaler.fit_transform(self.data[feature_columns])
            return self.data
        else:
            raise TypeError("Normalisasi hanya berlaku untuk data tabular.")

    def image_generator(self, target_size=(224, 224), batch_size=32, augment=False):
        """
        Membuat generator gambar untuk deep learning
        """
        if self.data_type == "image":
            if augment:
                datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )
            else:
                datagen = ImageDataGenerator(rescale=1./255)

            generator = datagen.flow_from_directory(
                self.data_path,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical'
            )
            return generator
        else:
            raise TypeError("Image generator hanya berlaku untuk data gambar.")

if __name__ == "__main__":
    # Contoh penggunaan
    # Muat CSV
    loader = DataLoader("dataset.csv", data_type="csv")
    df = loader.load_data()
    print("Data CSV:", df.head())

    # Split
    train_df, test_df = loader.split_data()
    print("Jumlah data latih:", len(train_df))
    print("Jumlah data uji:", len(test_df))

    # Normalisasi
    normalized_df = loader.normalize_data(feature_columns=["col1", "col2"])
    print("Data setelah normalisasi:", normalized_df.head())

    # Muat Gambar
    # img_loader = DataLoader("images/", data_type="image")
    # train_gen = img_loader.image_generator(augment=True)
