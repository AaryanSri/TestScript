from flask import Flask, request, jsonify, send_file
import os
from data_extraction import check_s3_bucket, download_s3_file, needs_update, update_info, run_sentiment_calculator, api_fetch_stock_data
from data_transformation import clean_data, add_data, pca_analysis, connect_to_writer_db, write_to_db, connect_to_reader_db, read_from_db
from io import BytesIO
app = Flask(__name__)

@app.route('/process', methods=['GET'])
def process_data():
    name = request.args.get('name')
    
    if not name:
        return jsonify({"error": "Parameter 'name' is required"}), 400

    try:
        # Execute functions from data_extraction.py
        print("hit")
        api_fetch_stock_data(name)
        update_needed = needs_update(name)
        update_info(update_needed, name)
        
        if update_needed:
            run_sentiment_calculator(name)
            # Execute functions from data_transformation.py
            filename = "Stocks_data/" + name + "_stock_data.csv"
            print("success")
            cleaned_df = clean_data(filename)
            print("success")
            upgraded_df = add_data(cleaned_df)
            print("success")
            pca_analysis_df = pca_analysis(upgraded_df)
            print("success")
            #update name to table name for Aurora
            name = name + "_DATA"
            writer_connection = connect_to_writer_db()
            print("success")
            print(pca_analysis_df.dtypes)
            write_to_db(pca_analysis_df, name, writer_connection)
            print("success")

        reader_connection = connect_to_reader_db()
        print("success")

        result = read_from_db(name, reader_connection)
        print("success")


        # Convert result to JSON serializable format if necessary
        # Assuming result is a DataFrame, convert it to JSON
        result_json = result.to_json(orient='records')
        tempfile = os.path.join(f'{name}.json')
        with open(tempfile, 'wb') as temp_file:
            temp_file.write(result_json.encode('utf-8'))
        
        response = send_file(tempfile, as_attachment=True, download_name=f'{name}.json', mimetype='application/json')
        @response.call_on_close
        def cleanup():
            try:
                os.remove(tempfile)
            except Exception as e:
                print(f"Error cleaning up temporary files: {e}")

        return response
    

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)