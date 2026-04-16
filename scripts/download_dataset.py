import os
from datasets import load_dataset
from rich.console import Console

def main():
    console = Console()
    dataset_name = "nhn309261/vietnamese-legal-documents"
    subset = "content"
    output_dir = "datasets/vietnamese-legal-content"

    # Create the target directory if it does not exist
    os.makedirs("datasets", exist_ok=True)
    
    console.print(f"[bold green]Downloading dataset {dataset_name} ({subset})...[/bold green]")
    try:
        # Based on count_sectors.py logic
        try:
            ds = load_dataset(dataset_name, name=subset, split="data")
        except ValueError:
            ds = load_dataset(dataset_name, name=subset, split="train")
            
    except Exception as e:
        console.print(f"[bold red]Failed to download dataset: {e}[/bold red]")
        return

    console.print(f"Successfully loaded {len(ds)} rows.")
    
    console.print(f"Saving to HF disk format at {output_dir}...")
    ds.save_to_disk(output_dir)
    
    parquet_path = "datasets/vietnamese_legal_content.parquet"
    console.print(f"Exporting to single Parquet file at {parquet_path}...")
    ds.to_parquet(parquet_path)
    
    console.print("[bold green]All done successfully![/bold green]")

if __name__ == "__main__":
    main()
