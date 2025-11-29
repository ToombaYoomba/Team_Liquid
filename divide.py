import pandas as pd
import os

def split_file_in_half(input_file, output_file):
    """Разделяет CSV файл пополам и сохраняет первую половину"""
    
    # Читаем файл
    df = pd.read_csv(input_file)
    
    # Вычисляем половину строк
    half_size = len(df) // 2
    
    # Берем первую половину
    first_half = df.head(half_size)
    
    # Сохраняем результат
    first_half.to_csv(output_file, index=False)
    print(f"Файл {input_file} разделен. Первая половина ({half_size} строк) сохранена в {output_file}")

def main():
    # Разделяем оба файла
    split_file_in_half('v2_hits_converted.csv', 'v2_hits_converted_first_half.csv')
    split_file_in_half('v2_visits_converted.csv', 'v2_visits_converted_first_half.csv')
    
    print("\nГотово! Оба файла разделены пополам.")

if __name__ == "__main__":
    main()