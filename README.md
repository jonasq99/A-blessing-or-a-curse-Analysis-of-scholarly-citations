# Run the code
1. Create a `.env` file in the `src` directory with the following content: 
   > OPENAI_API_KEY=<your_openai_api_key>
2. Choose an article and copy the filename including the file extension from the `all_data_articles` directory. 
   This is were the already scraped articles are stored. 
3. Run the following command in the terminal:
   ```bash
   python -m src.main <extraction-method> <filename>
   ```
   Replace `<extraction-method>` with the extraction method you want to use (anystyle, regex or tagger)
   Replace `<filename>` with the filename of the article you want to analyze. 
   For example:
   ```bash
   python -m src.main regex A_Colonial_Celebrity_in_the_New_Attention_Economy_Cecil_Rhodess_Cape-to-Cairo_Telegraph_and_Railway_Negotiations_in_1899.json
   ```
   This will generate a file in the `results/<extraction-method>` directory with the same name as the input file but with the `.csv` extension. 
   This file contains the results of the analysis, hence the footnote number, author, title and the prediction.

## Running individual scripts
Individual scripts should be executed with the -m option since the project uses relative imports.  
For example:
```bash
   python -m src.fewshot_cot_classification
   ```

## Prerequisites for Anystyle
1. Download and install Ruby. You can find more information for your operating system here: https://www.ruby-lang.org/en/downloads/. Usually, RubyGem should be automatically installed with Ruby. If not, please visit https://rubygems.org/pages/download for installation instructions.
2. Install Anystyle by executing:
    ```bash
    gem install anystyle
    ```
   In some cases, you may need admin rights to install Anystyle. In this case, you can use the following command (for Unix-based systems):
    ```bash
    sudo gem install anystyle
    ```

The Anystyle ruby script is automatically executed by the python code when the Anystyle method is chosen.
