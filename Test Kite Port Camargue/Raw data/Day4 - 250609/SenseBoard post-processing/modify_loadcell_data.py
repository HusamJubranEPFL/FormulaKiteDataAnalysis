import pandas as pd

def modify_loadcell_data(input_csv_path, output_csv_path):
    # Valeurs à ajouter à chaque LoadCell
    # LoadCell_1, LoadCell_2, LoadCell_3, LoadCell_4, LoadCell_5, LoadCell_6
    values_to_add = [2459, 4259, 4582, 816, 3802, 4256]

    try:
        # Lire le fichier CSV avec le séparateur point-virgule
        df = pd.read_csv(input_csv_path, sep=';')

        # Appliquer les modifications à chaque colonne LoadCell
        for i in range(1, 7):
            col_name = f'LoadCell_{i}'
            if col_name in df.columns:
                df[col_name] = df[col_name] + values_to_add[i-1]
            else:
                print(f"Avertissement : La colonne {col_name} n'a pas été trouvée dans le fichier CSV.")

        # Sauvegarder le fichier CSV modifié
        df.to_csv(output_csv_path, sep=';', index=False)
        print(f"Fichier modifié sauvegardé avec succès sous : {output_csv_path}")

    except FileNotFoundError:
        print(f"Erreur : Le fichier d'entrée '{input_csv_path}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    input_file = '_imu_log_cleaned.csv'
    output_file = '_imu_log_modified.csv'
    modify_loadcell_data(input_file, output_file)


