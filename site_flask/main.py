import json
import pandas as pd

from flask import Flask, request
from flask import render_template
from frames_tracker import predict


app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.hbs')


@app.route('/report', methods=['POST'])
def report():
    files = json.loads(request.data.decode('utf-8'))
    all_animals = predict('', files)
    name_folders = []
    classes = []
    date_registration_starts = []
    date_registration_ends = []
    counts = []

    for row in all_animals:
        name_folders.append(int(row['filename'].split('/')[-2]))
        classes.append(row['cls'])
        date_registration_starts.append(str(row['first_seen']))
        date_registration_ends.append(str(row['last_seen']))
        counts.append(row['count'])

    dataframe_data = {
        'name_folder': name_folders,
        'class': classes,
        'date_registration_start': date_registration_starts,
        'date_registration_end': date_registration_ends,
        'count': counts
    }
    df = pd.DataFrame(dataframe_data)
    df.to_csv('predict.csv', index=False)
    # print(all_animals)
    # print(type(all_animals))
    return all_animals


if __name__ == '__main__':
    app.run(debug=False, port=5001)
