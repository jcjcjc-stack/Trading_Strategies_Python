import re


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(value, label):
    if not IDENTIFIER_PATTERN.match(value):
        raise ValueError(f"Invalid WRDS {label}: {value}")

    return value


def connect_wrds(**kwargs):
    try:
        import wrds
    except ImportError as error:
        raise ImportError(
            "Install wrds before using WRDS research data: pip install wrds"
        ) from error

    return wrds.Connection(**kwargs)


def load_wrds_table(
    library,
    table,
    columns=None,
    obs=None,
    date_cols=None,
    **connection_kwargs,
):
    library = validate_identifier(library, "library")
    table = validate_identifier(table, "table")

    with connect_wrds(**connection_kwargs) as db:
        return db.get_table(
            library=library,
            table=table,
            columns=columns,
            obs=obs,
            date_cols=date_cols,
        )


def query_wrds(sql, params=None, date_cols=None, **connection_kwargs):
    with connect_wrds(**connection_kwargs) as db:
        return db.raw_sql(
            sql,
            params=params,
            date_cols=date_cols,
        )


def load_fama_french_factors(
    table="factors_daily",
    library="ff",
    columns=None,
    start=None,
    end=None,
):
    date_column = "date"
    library = validate_identifier(library, "library")
    table = validate_identifier(table, "table")

    selected_columns = columns or ["date", "mktrf", "smb", "hml", "rf"]
    for column in selected_columns:
        validate_identifier(column, "column")

    where_clauses = []
    params = {}

    if start:
        where_clauses.append(f"{date_column} >= %(start)s")
        params["start"] = start

    if end:
        where_clauses.append(f"{date_column} <= %(end)s")
        params["end"] = end

    where_sql = f"where {' and '.join(where_clauses)}" if where_clauses else ""
    sql = f"""
        select {", ".join(selected_columns)}
        from {library}.{table}
        {where_sql}
        order by {date_column}
    """

    return query_wrds(sql, params=params, date_cols=[date_column])
