# openalex.py
import os
import requests
import json
import argparse
from collections import defaultdict
from scipy.stats import percentileofscore
from datetime import datetime, date

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path",
    type=str,
    help="Absolute path to save the data",
    default="./data",
)
parser.add_argument(
    "--percentage",
    type=float,
    help="Percentage of top and bottom citations of papers to include",
    # required=True,
    default=1,
)
parser.add_argument(
    "--journal_list",  # The name of the argument
    nargs="+",  # Accept one or more arguments
    type=str,  # Specify the type of each item in the list
    # required=True,  # Make the argument mandatory
    help="A list of journals to include",
    default=["Educational Psychologist"],
)
parser.add_argument(
    "--year_start", type=int, help="Start year for the data", 
    # required=True, 
    default=2018
)
parser.add_argument("--year_end", type=int, help="End year for the data", 
                    # required=True, 
                    default=2020)
parser.add_argument(
    "--years_after",
    type=int,
    help="The impact of the paper will be calculated years_after years after the publication year",
    # required=True,
    default=2,
)
args = parser.parse_args()
args_dict = vars(args)

for key, value in args_dict.items():
    print(f"{key}: {value}")

def get_source_id(journal_name):
    url = f"https://api.openalex.org/sources?filter=display_name.search:{journal_name}"
    response = requests.get(url)
    try:
        results = response.json()["results"]
    except:
        # raise Exception("No valid response")
        print(f"No valid response for {journal_name}")
        return -1
    for journal in results:
        if journal["display_name"] == journal_name:
            return journal["id"].split("/")[-1]
    # No search results
    return -1


def add_percentile_fields(data, years_after: int, percentage: int):
    current_year = datetime.now().year  # Get the current year
    # Map publication years to lists of citation counts
    citations_by_pub_year = defaultdict(list)
    paper_citations_by_id = {}

    # First pass: collect citation counts for the specified year after publication
    for paper in data:
        pub_year = paper["publication_year"]
        counts_by_year = paper.get("counts_by_year", [])
        # Create a dictionary of year to cited_by_count
        year_citation_dict = {
            item["year"]: item["cited_by_count"] for item in counts_by_year
        }
        # Get the citation count for the specified year after publication
        citation_count = year_citation_dict.get(pub_year + years_after, 0)
        # Store the citation count for this paper
        paper_citations_by_id[paper["id"]] = citation_count
        # Add the citation count to the list for this publication year
        citations_by_pub_year[pub_year].append(citation_count)

    # Second pass: compute and add percentile ranks for the specified year
    for paper in data:
        pub_year = paper["publication_year"]
        citation_count = paper_citations_by_id[paper["id"]]
        citations_list = citations_by_pub_year[pub_year]
        # Calculate the percentile rank using scipy.stats.percentileofscore
        percentile = percentileofscore(citations_list, citation_count, kind="strict") # A percentileofscore of 80% means that 80% of values are less than the provided score.
        # Add the new percentile fields to the paper
        field_name1 = f"citation_{years_after}_years_after_percentile"
        # Set to 'null' if pub_year + years_after exceeds the current year
        if pub_year + years_after > current_year:
            paper[field_name1] = paper["high_impact"] = paper["low_impact"] = None
        else:
            paper[field_name1] = percentile
            # citation_{years_after}_years_after_is_top_{percentage}
            paper["high_impact"] = int(percentile > (100 - percentage))
            # citation_{years_after}_years_after_is_bottom_{percentage}
            paper["low_impact"] = int(percentile < percentage)


def get_journal_data(journal_id, year):
    url = f"https://api.openalex.org/works?select=id,title,abstract_inverted_index,publication_year,counts_by_year,type,biblio&filter=locations.source.id:{journal_id},publication_year:{year}&per-page=200&cursor=*"
    i = 0
    data_all = []
    while True:
        response = requests.get(url)
        if not response:
            # raise Exception("No valid response")
            print(f"No valid response for {journal_id}, {year}, page {i}")
            return []
        data = response.json()
        data_all.extend(data["results"])
        # Check for the next page link
        cursor = data["meta"].get("next_cursor")
        print(f"Downloading {journal_id}, {year}, page {i}")
        if not cursor or not data["results"]:
            break
        elif cursor:
            url = f"https://api.openalex.org/works?select=id,title,abstract_inverted_index,publication_year,counts_by_year,type,biblio&filter=locations.source.id:{journal_id},publication_year:{year}&per-page=200&cursor={cursor}"
            i += 1
        else:
            break
    return data_all


def reconstruct_abstract(abstract_inverted_index):
    # Determine the maximum position
    positions = []
    for pos_list in abstract_inverted_index.values():
        positions.extend(pos_list)
    if positions == []:
        return None
    max_pos = max(positions)
    if max_pos < 40:
        # Only keep papers with abstract length > 40
        return None
    # Initialize a list to hold words at their positions
    words_list = [""] * (max_pos + 1)

    # Assign words to their corresponding positions
    for word, pos_list in abstract_inverted_index.items():
        for pos in pos_list:
            words_list[pos] = word

    # Join the words to form the abstract text
    abstract_text = " ".join(words_list)
    if abstract_text.endswith("..."):
        return None
    return abstract_text


def is_valid_page_range(first_page, last_page):
    """Validate and calculate the number of pages based on first_page and last_page."""
    try:
        # Convert to integers and calculate page count
        first_page = int(first_page)
        last_page = int(last_page)
        return last_page - first_page + 1  # Inclusive range
    except (ValueError, TypeError):
        # Handle non-numeric or missing values
        return -1  # Invalid page count


def is_valid_page_range(first_page, last_page):
    """Validate and calculate the number of pages based on first_page and last_page."""
    try:
        # Convert to integers and calculate page count
        first_page = int(first_page)
        last_page = int(last_page)
        return last_page - first_page + 1  # Inclusive range
    except (ValueError, TypeError):
        # Handle non-numeric or missing values
        return -1  # Invalid page count


def download_and_process_data(
    journal_name, journal_id, pub_year, percentage, years_after, save_path
):
    if journal_id == -1:
        # no corresponding journal
        return 0, 0
    # Fetch journal data
    data = get_journal_data(journal_id, pub_year)
    print(f"Got raw data for journal '{journal_name}' ({journal_id}) for year {pub_year}")

    processed_data = []
    for paper in data:
        # Filter for type 'article' and presence of abstract
        if (
            paper.get("type") != "article"
            or paper.get("abstract_inverted_index") is None
        ):
            continue

        # Reconstruct abstract and filter by length
        paper["abstract"] = reconstruct_abstract(paper["abstract_inverted_index"])
        if not paper["abstract"]:
            continue

        processed_data.append(paper)

    # Add percentile fields (assumes this modifies the data in place)
    add_percentile_fields(processed_data, years_after, percentage)

    # Separate high and low impact papers
    high_impacts = [paper for paper in processed_data if paper["high_impact"] == 1]
    low_impacts = [paper for paper in processed_data if paper["low_impact"] == 1]

    # Determine the minimum count to balance high and low impacts
    min_count = min(len(high_impacts), len(low_impacts))

    if min_count == 0:
        print(f"{journal_name}, {pub_year}, {percentage}%: Insufficient data. No result can be returned for this journal.")
        return 0, 0
    else:
        # Sort high_impacts by descending percentile and keep top min_count
        high_impacts_sorted = sorted(
            high_impacts,
            key=lambda x: x[f"citation_{years_after}_years_after_percentile"],
            reverse=True,
        )[:min_count]

        # Sort low_impacts by ascending percentile and keep top min_count
        low_impacts_sorted = sorted(
            low_impacts,
            key=lambda x: x[f"citation_{years_after}_years_after_percentile"],
        )[:min_count]

        # Combine the balanced high and low impacts
        balanced_data = high_impacts_sorted + low_impacts_sorted

    # Prepare the final_data with relevant fields
    final_data = []
    for paper in balanced_data:
        final_paper = {
            "id": paper["id"],
            "title": paper["title"],
            "abstract": paper["abstract"],
            "high_impact": paper["high_impact"],
            # "low_impact": paper["low_impact"],
            # Uncomment the following line if you want to include percentile in the output
            # f"citation_{years_after}_years_after_percentile": paper[f"citation_{years_after}_years_after_percentile"],
        }
        final_data.append(final_paper)

    # Check if the folder exists; if not, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare file path
    file_path = os.path.join(save_path, f"{journal_name}_{pub_year}.json")

    # Write processed data to a file in the folder
    with open(file_path, "w") as file:
        json.dump(final_data, file, indent=4)

    print(f"Processed data for {journal_name} ({pub_year}) saved to {file_path}.")
    print(f"Number of high_impact and low_impact papers: {min_count}")

    # Return the count of processed papers
    return len(data), len(final_data)

if __name__ == "__main__":
    # Prepare metadata for this experiment
    metadata = {
        "meta": {
            "journals": args_dict["journal_list"],
            "year_start": args_dict["year_start"],
            "year_end": args_dict["year_end"],
            "percentage": args_dict["percentage"],
            "years_after": args_dict["years_after"],
            "save_path": f"{args_dict["save_path"]}",  # naming convention allows "+" 
            "download_date": date.today().strftime("%Y-%m-%d"),
            "final_data_count_split": {},
            "filtered_data_count_split": {},
        }
    }
    if not os.path.exists(os.path.join(args_dict["save_path"], "metadata.json")):
        total_data_count = 0
        for journal in args_dict["journal_list"]:
            journal_id = get_source_id(journal)
            # Initialize the count for this journal
            metadata["meta"]["final_data_count_split"][journal] = {}
            metadata["meta"]["filtered_data_count_split"][journal] = {}
            for year in range(args_dict["year_start"], args_dict["year_end"] + 1):
                raw_data_count, final_data_count = download_and_process_data(
                    journal,
                    journal_id,
                    year,
                    args_dict["percentage"],
                    args_dict["years_after"],
                    args_dict["save_path"],
                )
                metadata["meta"]["filtered_data_count_split"][journal][year] = raw_data_count - final_data_count
                metadata["meta"]["final_data_count_split"][journal][year] = final_data_count
                total_data_count += final_data_count
        metadata["meta"]["total_data_count"] = total_data_count
            


        # After processing all data, save the combined metadata
        metadata_file_path = os.path.join(args_dict["save_path"], "metadata.json")
        with open(metadata_file_path, "w") as file:
            json.dump(metadata, file, indent=4)
        print(f"Combined metadata saved to {metadata_file_path}.")
    # process data into huggingface dataset
    os.makedirs(f"{args_dict['save_path']}/huggingface", exist_ok=True)
    """
    huggingface data are a set of keys, each key is a feature name, and each value is a list of feature values. 
    """
    data_dict = {}
    data_dict["year"] = []
    for journal in args_dict["journal_list"]:
        for year in range(args_dict["year_start"], args_dict["year_end"] + 1):
            file_path = os.path.join(args_dict["save_path"], f"{journal}_{year}.json")
            with open(file_path, "r") as file:
                data = json.load(file)
                for paper in data:
                    for key, value in paper.items():
                        if key not in data_dict:
                            data_dict[key] = []
                        data_dict[key].append(value)
                    data_dict["year"].append(year)
    # change key name: "high_impact" -> "label"
    data_dict["label"] = data_dict.pop("high_impact")
    # shuffle the dataset but preserve relative list ordering
    indices = list(range(len(data_dict["label"])))
    import random
    random.shuffle(indices)
    for key, value in data_dict.items():
        data_dict[key] = [value[i] for
                          i in indices]
    # Save the huggingface dataset
    train, val, test = 0.8, 0.1, 0.1
    train_size = int(len(data_dict["label"]) * train)
    val_size = int(len(data_dict["label"]) * val)
    test_size = len(data_dict["label"]) - train_size - val_size
    data_dict_train = {key: value[:train_size] for key, value in data_dict.items()}
    data_dict_val = {key: value[train_size:train_size + val_size] for key, value in data_dict.items()}
    data_dict_test = {key: value[train_size + val_size:] for key, value in data_dict.items()}
    with open(f"{args_dict['save_path']}/huggingface/citation_train.json", "w") as file:
        json.dump(data_dict_train, file, indent=4)
    with open(f"{args_dict['save_path']}/huggingface/citation_val.json", "w") as file:
        json.dump(data_dict_val, file, indent=4)
    with open(f"{args_dict['save_path']}/huggingface/citation_test.json", "w") as file:
        json.dump(data_dict_test, file, indent=4)
    print(f"Saved huggingface dataset to {args_dict['save_path']}/huggingface/citation_train.json, citation_val.json, citation_test.json")


