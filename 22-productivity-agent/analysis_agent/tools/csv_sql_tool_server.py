import pandas as pd
from mcp.server.fastmcp import FastMCP
from pandasql import sqldf

mcp = FastMCP("csv-sql-runner")

@mcp.tool()
def run_csv_sql(customer_id: str) -> str:
    """
    This tool is used to query Customers database. Returns the details about given customer_id
    """

    # Load CSV
    df = pd.read_csv("./sample_customers.csv")
    
    # Define query
    query = """
    SELECT *
    FROM df
    WHERE customer_id = 
    """ + customer_id
    
    result = sqldf(query)
    return str(result)

if __name__ == "__main__":
    mcp.run(transport="stdio")