# Import required libraries
import json  # For reading and writing JSON files
import requests  # For making HTTP requests to APIs
from typing import List, Dict, Optional  # For type hinting
from PyPDF2 import PdfReader  # For extracting text from PDF resumes
from fastmcp import FastMCP  # For MCP server functionality
from pathlib import Path  # For file and directory path management
import os  # For environment variable access
from dotenv import load_dotenv  # For loading .env files


# === Setup Directories ===
# Set up base directory and subdirectories for job data and resumes
BASE_DIR = Path(".")  # Current directory as base
JOBS_DIR = BASE_DIR / "jobs" / "saved_by_candidate"  # Directory to save candidate's jobs
TEMP_DIR = BASE_DIR / "jobs" / "temp"  # Temporary directory for fetched jobs
RESUME_PATH = BASE_DIR / "resume" / "resume.pdf"  # Path to the candidate's resume
# Create directories if they don't exist
JOBS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# === Load .env from parent directory ===
# Load environment variables from a .env file located in the parent directory
PARENT_DIR = BASE_DIR.parent
dotenv_path = PARENT_DIR / ".env"
load_dotenv(dotenv_path)


# === JSearch API Info ===
# Get API credentials from environment variables
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")  # Your RapidAPI key
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")  # JSearch API host


# === Init MCP ===
# Initialize the FastMCP server with a name
mcp = FastMCP("Job Market Explorer")

# === Tool: Search Jobs ===
# Tool: Defines a function to search for jobs using the JSearch API and returns summarized job info.

@mcp.tool()
def search_jobs(role: str, location: str, max_results: int = 5) -> List[Dict]:
    """
    Fetch jobs using JSearch API and store them temporarily. Return key info.

    Args:
        role: The role to search for.
        location: The location to search for.
        max_results: The maximum number of jobs to return.

    Returns:
        A list of dictionaries containing the job information.
    """
    # Step 1: Prepare API request headers with credentials
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    # Step 2: Build the search query and URL
    query = f"{role} in {location}"
    url = f"https://{RAPIDAPI_HOST}/search?query={query}&num_pages=1"

    # Step 3: Make the API request to fetch jobs
    response = requests.get(url, headers=headers)
    # Step 4: Parse the JSON response
    data = response.json()
    # Step 5: Extract the job list and limit to max_results
    job_list = data.get("data", [])[:max_results]

    # Step 6: Handle case where no jobs are found
    if not job_list:
        return [{"message": "No jobs found."}]

    # Step 7: Save fetched jobs to a temporary file for later use
    temp_path = TEMP_DIR / "fetched_jobs_temp.json"
    with open(temp_path, "w") as f:
        json.dump(job_list, f, indent=2)

    # Step 8: Summarize job info for each job
    results = []
    for job in job_list:
        desc = job.get("job_description", "")
        # If description is a string, summarize it (first 1000 and last 1000 chars if long)
        if isinstance(desc, str):
            if len(desc) <= 1000:
                summary = desc
            else:
                summary = desc[:1000] + ("\n...\n" if len(desc) > 2000 else "") + desc[-1000:]
        else:
            summary = ""

        # Collect key job info in a dictionary
        results.append({
            "job_id": job.get("job_id"),
            "title": job.get("job_title"),
            "company": job.get("employer_name"),
            "location": job.get("job_city"),
            "description": summary,
            "apply_link": job.get("job_apply_link", "Not provided")
        })

    # Step 9: Return the summarized job info list
    return results


# === Tool: Save Job ===
# Tool: Saves a selected job from the fetched jobs list into the candidate's saved folder, including salary info.


@mcp.tool()
def save_job(job_id: str, salary: Optional[str] = None) -> str:
    """
    Save a specific job from temporary list into candidate's saved folder.
    If salary is not provided, it tries to extract it from the fetched job data.

    Args:
        job_id: The ID of the job to save.
        salary (optional): The salary of the job to save.

    Returns:
        A string indicating the job was saved successfully.
    """
    # Step 1: Define the path to the temporary jobs file
    temp_path = TEMP_DIR / "fetched_jobs_temp.json"
    # Step 2: Check if the temp file exists; if not, instruct user to run search first
    if not temp_path.exists():
        return "No fetched jobs available. Please run search_jobs first."

    # Step 3: Load the list of fetched jobs from the temp file
    with open(temp_path, "r") as f:
        jobs = json.load(f)

    # Step 4: Find the job with the specified job_id
    selected = next((job for job in jobs if job.get("job_id") == job_id), None)
    if not selected:
        return f"Job ID {job_id} not found in fetched data."

    # Step 5: Prefer the LLM-provided salary if available
    final_salary = salary

    # Step 6: If salary not provided, try to extract from job data fields
    if not final_salary:
        currency = selected.get("salary_currency")
        min_base = selected.get("min_base_salary") or selected.get("job_min_salary")
        max_base = selected.get("max_base_salary") or selected.get("job_max_salary")
        min_add = selected.get("min_additional_pay")
        max_add = selected.get("max_additional_pay")
        salary_period = selected.get("job_salary_period")

        # If all required fields are present, calculate total salary range
        if currency and min_base and max_base:
            total_min = int(min_base + (min_add or 0))
            total_max = int(max_base + (max_add or 0))
            per = f" per {salary_period.lower()}" if salary_period else ""
            final_salary = f"{currency} {total_min:,} – {total_max:,}{per}"

    # Step 7: If salary still not found, mark as not specified
    if not final_salary:
        final_salary = "Not specified"

    # Step 8: Compose the job data dictionary to save
    job_data = {
        "title": selected.get("job_title", "Not specified"),
        "company": selected.get("employer_name", "Not specified"),
        "location": selected.get("job_city", "Not specified"),
        "description": selected.get("job_description", "Not specified"),
        "employment_type": selected.get("job_employment_type", "Not specified"),
        "posted_at": selected.get("job_posted_at_datetime_utc", "Not specified"),
        "apply_link": selected.get("job_apply_link", "Not specified"),
        "salary": final_salary,
    }

    # Step 9: Save the job data to a file in the saved_by_candidate directory
    role_folder = JOBS_DIR / "general"
    role_folder.mkdir(exist_ok=True)
    job_file = role_folder / f"{job_id}.json"
    with open(job_file, "w") as f:
        json.dump(job_data, f, indent=2)

    # Step 10: Return a success message
    return f"Job {job_id} saved successfully with salary: {final_salary}"



# === Resource: Resume ===
# Resource: Extracts and returns the text content from the candidate's resume PDF as markdown.


@mcp.resource("resume://default")
def candidate_resume() -> str:
    """
    Extract text from resume.pdf and return as markdown.
    """
    try:
        # Step 1: Open and read the resume PDF file
        reader = PdfReader(str(RESUME_PATH))
        # Step 2: Extract text from each page and join with double newlines
        text = "\n\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        # Step 3: Return the extracted text as markdown, or a message if empty
        return f"# Resume Content\n\n{text.strip() or 'No readable content.'}"
    except Exception as e:
        # Step 4: Handle errors (e.g., file not found, unreadable PDF)
        return f"Error reading resume: {e}"



# === Resource: Saved Jobs ===
# Resource: Returns a markdown listing of all jobs saved by the candidate.


@mcp.resource("jobs://saved")
def get_saved_jobs() -> str:
    """
    Return markdown listing of all saved jobs.
    """
    # Step 1: Start the markdown content for saved jobs
    content = "# Saved Jobs\n\n"
    # Step 2: Loop through all saved job JSON files
    for file in JOBS_DIR.glob("**/*.json"):
        try:
            # Step 3: Open and load each job file
            with open(file, "r") as f:
                job = json.load(f)
                # Step 4: Append job details to the markdown content
                content += f"## {job.get('title', 'Untitled')}\n"
                content += f"- **Company**: {job.get('company')}\n"
                content += f"- **Location**: {job.get('location')}\n"
                content += f"- **Description**: {job.get('description')}\n"
                content += f"- **Employment Type**: {job.get('employment_type')}\n"
                content += f"- **Apply**: [Link]({job.get('apply_link')})\n"
                content += f"- **Salary**: {job.get('salary')}\n\n"
        except Exception:
            # Step 5: Skip files that can't be read or parsed
            continue

    # Step 6: If no jobs found, return a message; otherwise, return the content
    if content.strip() == "# Saved Jobs":
        return "# No saved jobs found."
    return content



# === Prompt: Market Analysis ===
# Prompt: Generates instructions for analyzing the job market for a given role and location.
@mcp.prompt()
def analyze_job_market(role: str, location: str, num_jobs: int = 5) -> str:
    """
    Analyze the job market for top {num_jobs} jobs for '{role}' in '{location}'.
    """

    # Step 1: Return a prompt string with instructions for job market analysis
    return f"""Analyze the job market for top {num_jobs} jobs for '{role}' in '{location}'.

            Steps:
            1. Run job search mcp tool with suitable roles and locations.
            2. Review fields like title, company, type, and description.
            3. Summarize:
            - Most common roles
            - Repeated skills or keywords
            - Salary trends (if any)
            - Remote vs onsite distribution

            Structure insights clearly in markdown format."""



# === Prompt: Personalized Job Recommender ===
# Prompt: Generates instructions to recommend jobs based on the candidate's resume and preferences.

@mcp.prompt()
def personalized_job_recommender() -> str:
    """
    Use the resume to extract key skills, interests, and preferred job types.
    """

    # Step 1: Return a prompt string with instructions for personalized job recommendations
    return """Use the resume to extract key skills, interests, and preferred job types.
            Then:
            1. Call job search mcp tool with suitable roles and locations.
            2. Review descriptions and recommend jobs.
            3. Optionally call save job mcp tool on top matches.

            Output sections:
            - Top Matches
            - Stretch Roles
            - Company Highlights
            """



# Prompt: Generates instructions to create a summary report matching the resume to saved jobs.
@mcp.prompt()
def create_match_report() -> str:
    """
    Given the attached jobs data and resume, create a concise and accurate summary of how well the resume matches the jobs.
    """

    # Step 1: Return a prompt string with instructions for creating a match report
    return """Given the attached jobs data and resume, create a concise and accurate summary of how well the resume matches the jobs.

Output sections:
- Job Summary
- Resume Summary
- Job Match Summary
"""



# === Run MCP ===
# If this script is run directly, start the MCP server using stdio transport
if __name__ == "__main__":
    mcp.run(transport="stdio")
