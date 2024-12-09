import calendar
import enum
import io
import json
import logging
import os
import os.path
import pytz
import re
import time
import traceback
import urllib.parse
from datetime import datetime, timedelta
from itertools import cycle
from pathlib import Path
from random import randint

import pandas as pd
import requests
from bs4 import BeautifulSoup

#from .config import CONTRACT_MAP, EXCHANGES


logger = logging.getLogger(__name__)


class HistoricalDataResult(enum.Enum):
    NONE = 1
    OK = 2
    EXISTS = 3
    EXCEED = 4
    INSUFFICIENT = 5


class Resolution(enum.Enum):
    Day = (1, "daily")
    Hour = (2, "hourly")

    def __init__(self, value, adjective):
        self._value_ = value
        self._adjective_ = adjective

    @property
    def adj(self):
        return self._adjective_


class BCException(Exception):
    pass


class IntegrityException(Exception):
    pass


class RecentUpdateException(Exception):
    pass


class EmptyDataException(Exception):
    pass


MONTH_LIST = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
BARCHART_URL = "https://www.barchart.com/"

def create_bc_session(barchart_username, barchart_password, do_login=True):
    """
    
    Create and return a web session, optionally logging into Barchart with the supplied
    credentials.

    Args:
        barchart_username: The username for Barchart login.
        barchart_password: The password for Barchart login.
        do_login: If True, authenticate session with Barchart credentials.

    Returns:
        A requests.Session instance.

    Raises:
        Exception: If credentials are invalid.
    """

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    if do_login:
        # GET the login page, scrape to get CSRF token
        resp = session.get(BARCHART_URL + "login")
        soup = BeautifulSoup(resp.text, "html.parser")
        tag = soup.find(type="hidden")
        csrf_token = tag.attrs["value"]
        logger.info(
            f"GET {BARCHART_URL + 'login'}, status: {resp.status_code}, "
            f"CSRF token: {csrf_token}"
        )

        # login to site with a POST
        payload = {
            "email": barchart_username,
            "password": barchart_password,
            "_token": csrf_token,
        }
        resp = session.post(BARCHART_URL + "login", data=payload)
        logger.info(f"POST {BARCHART_URL + 'login'}, status: {resp.status_code}")
        if resp.url == BARCHART_URL + "login":
            raise BCException("Invalid credentials")

    return session


def save_prices_for_contract(
    session: requests.Session,
    contract: str,
    save_path: str,
    start_date: datetime,
    end_date: datetime,
    dry_run: bool = False,
):
    """
    Save prices for an individual futures contract.

    Args:
        session: requests.Session instance
        contract: Barchart style contract identifier, eg GCH24 for March 2024 Gold
        save_path: full path where price file will be saved
        start_date: start date
        end_date: end date
        dry_run: if True, provides useful diagnostic info but does not execute

    Returns:
        A HistoricalDataResult instance, representing the result of the operation
    """

    res = _get_resolution(save_path)

    try:
        # do we have this file already?
        if os.path.isfile(save_path):
            logger.info(
                f"{res.adj} data for contract '{contract}' already downloaded "
                f"({save_path}) - skipping\n"
            )
            return HistoricalDataResult.EXISTS

        if _insufficient_data(session, contract, res):
            logger.info(f"Insufficient {res.adj} data for '{contract}' - skipping\n")
            return HistoricalDataResult.INSUFFICIENT

    except Exception as e:  # skipcq broad by design
        logger.error(f"Problem: {e}, {traceback.format_exc()}")

    logger.info(
        f"getting historic {res.adj} prices for contract '{contract}', "
        f"from {start_date.strftime('%Y-%m-%d')} "
        f"to {end_date.strftime('%Y-%m-%d')}"
    )

    try:
        # open historic data download page for required contract
        url = f"{BARCHART_URL}futures/quotes/{contract}/historical-download"
        hist_resp = session.get(url)
        logger.info(f"GET {url}, status {hist_resp.status_code}")

        if hist_resp.status_code != 200:
            logger.info(f"No downloadable data found for contract '{contract}'\n")
            return HistoricalDataResult.NONE

        xsrf = urllib.parse.unquote(hist_resp.cookies["XSRF-TOKEN"])

        # scrape page for csrf_token
        hist_soup = BeautifulSoup(hist_resp.text, "html.parser")
        hist_tag = hist_soup.find(name="meta", attrs={"name": "csrf-token"})
        hist_csrf_token = hist_tag.attrs["content"]

        # check allowance
        payload = {"onlyCheckPermissions": "true"}
        headers = {
            "content-type": "application/x-www-form-urlencoded",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": url,
            "x-xsrf-token": xsrf,
        }
        resp = session.post(BARCHART_URL + "my/download", headers=headers, data=payload)

        allowance = json.loads(resp.text)

        if allowance.get("error") is not None:
            return HistoricalDataResult.EXCEED

        if allowance["success"]:
            logger.info(
                f"POST {BARCHART_URL + 'my/download'}, "
                f"status: {resp.status_code}, "
                f"allowance success: {allowance['success']}, "
                f"allowance count: {allowance['count']}"
            )

            # download data
            xsrf = urllib.parse.unquote(resp.cookies["XSRF-TOKEN"])
            headers = {
                "content-type": "application/x-www-form-urlencoded",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": url,
                "x-xsrf-token": xsrf,
            }

            payload = {
                "_token": hist_csrf_token,
                "fileName": contract + "_Daily_Historical Data",
                "symbol": contract,
                "fields": "tradeTime.format(Y-m-d),openPrice,highPrice,lowPrice,"
                "lastPrice,volume",
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
                "orderBy": "tradeTime",
                "orderDir": "asc",
                "method": "historical",
                "limit": "20000",
                "customView": "true",
                "pageTitle": "Historical Data",
            }

            dateformat = "%m/%d/%Y %H:%M"
            if res == Resolution.Day:
                payload["type"] = "eod"
                payload["period"] = "daily"
                dateformat = "%Y-%m-%d"

            elif res == Resolution.Hour:
                payload["type"] = "minutes"
                payload["interval"] = 60

            if not dry_run:
                resp = session.post(
                    BARCHART_URL + "my/download", headers=headers, data=payload
                )
                logger.info(
                    f"POST {BARCHART_URL + 'my/download'}, "
                    f"status: {resp.status_code}, "
                    f"data length: {len(resp.content)}"
                )
                if resp.status_code == 200:
                    if "Error retrieving data" not in resp.text:
                        iostr = io.StringIO(resp.text)
                        df = pd.read_csv(iostr, skipfooter=1, engine="python")
                        df["Time"] = pd.to_datetime(df["Time"], format=dateformat)
                        df.set_index("Time", inplace=True)
                        df.index = df.index.tz_localize(tz="US/Central").tz_convert(
                            "UTC"
                        )
                        df = df.rename(columns={"Last": "Close"})

                        logger.info(f"writing to: {save_path}")
                        df.to_csv(save_path, date_format="%Y-%m-%dT%H:%M:%S%z")

                    else:
                        logger.info(
                            f"Barchart data problem for '{contract}', not writing"
                        )
            else:
                logger.info(f"Not POSTing to {BARCHART_URL + 'my/download'}, dry_run")

            logger.info(
                f"Finished getting Barchart historic {res.adj} prices for {contract}\n"
            )

        return HistoricalDataResult.OK

    except Exception as e:  
        logger.error(f"Error {e}")


def get_barchart_downloads(
    session: requests.Session,
    contract_map: dict = None,
    contract_list: list = None,
    instr_list: list = None,
    save_dir: str = None,
    start_year: int = 1950,
    end_year: int = 2025,
    dry_run: bool = False,
    do_daily: bool = True,
    pause_between_downloads: bool = True,
    default_day_count: int = 400,
):
    """
    Run a download session, performing as many contract downloads as possible, given
    the config, parameters, existing files, and available daily allowance.

    Args:
        session: requests.Session instance
        contract_map: dict containing instrument config
        contract_list: optional list of Barchart contract IDs we want to download in
            this run. If provided, `start_year` and `start_year` are ignored. If not
            provided, a list will be created based on the parameters. See
            `_build_contract_list()`
        instr_list: list of instrument codes (eg GOLD, AUD) we want to download in this
            run
        save_dir: full path to the directory where we want downloaded files to be saved
        start_year: start year as an int
        end_year: end year as an int
        dry_run: if True, provides useful diagnostic info but does not execute
        do_daily: if True, download daily as well as hourly price files
        pause_between_downloads: if True, wait a random short period between downloads
        default_day_count: default number of days of data to download
    """
    if contract_map is None:
        contract_map = CONTRACT_MAP

    inv_contract_map = _build_inverse_map(contract_map)

    max_exceeded = False

    try:
        if contract_list is None:
            contract_list = _build_contract_list(
                start_year, end_year, instr_list=instr_list, contract_map=contract_map
            )

        for contract in contract_list:
            if max_exceeded:
                break

            for resolution in Resolution if do_daily else [Resolution.Hour]:
                # work out instrument code and get config
                market_code = contract[: len(contract) - 3]
                instr_code = inv_contract_map[market_code.upper()]
                instr_config = contract_map[instr_code]

                # get contract month and year
                month, year = _get_contract_month_year(contract)

                # build save path
                save_path = _build_save_path(
                    instr_code, month, year, resolution, save_dir
                )

                # calculate date range
                start_date, end_date = _get_start_end_dates(
                    month,
                    year,
                    instr_config,
                    default_day_count=default_day_count,
                )

                if _before_available_res(resolution, start_date, instr_config):
                    date_type = "tick" if resolution == Resolution.Hour else "EOD"
                    logger.info(
                        f"{resolution.adj} prices for {contract} starting "
                        f"{start_date.strftime('%Y-%m-%d')} is before configured "
                        f"{date_type} date - skipping\n"
                    )
                    continue

                # download and save
                result = save_prices_for_contract(
                    session,
                    contract,
                    save_path,
                    start_date,
                    end_date,
                    dry_run=dry_run,
                )

                if result in [
                    HistoricalDataResult.EXISTS,
                    HistoricalDataResult.NONE,
                    HistoricalDataResult.INSUFFICIENT,
                ]:
                    continue
                elif result == HistoricalDataResult.EXCEED:
                    logger.info("Max daily download reached, aborting")
                    max_exceeded = True
                    break
                else:
                    if pause_between_downloads:
                        # cursory attempt to not appear like a bot
                        time.sleep(0 if dry_run else randint(7, 15))

        # logout
        resp = session.get(BARCHART_URL + "logout", timeout=10)
        logger.info(f"GET {BARCHART_URL + 'logout'}, status: {resp.status_code}")

    except Exception as e:  # skipcq broad by design
        logger.error(f"Error {e}")
        traceback.print_exc()


def update_barchart_downloads(
    instr_code: str = "GOLD",
    contract_map: dict = None,
    save_dir: str = None,
    days_ago: int = 360,
    dry_run: bool = False,
    split_freq: bool = True,
):
    """
    Update recent previously downloaded files for an instrument.

    Considers previously downloaded contract files where the contract date is more
    recent than `days_ago`. For each file, will update it with any new price data rows,
    given the existing resolution.

    Args:
        instr_code: instrument code (eg GOLD)
        contract_map: dict containing instrument config
        save_dir: full path to the directory where previously downloaded files are
            located
        days_ago: how many days to look back. A file's contract date is assumed to be
            the 1st of the month. So GCH23 would be 1st March 2023
        dry_run: if True, provides useful diagnostic info but does not execute
        split_freq: True if we are expecting to find split frequency files
    """
    if contract_map is None:
        contract_map = CONTRACT_MAP

    from_date = datetime.now() - timedelta(days=days_ago)

    logger.info(f"Updating contract prices for {instr_code}")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    check_integrity_list = []

    file_names = _get_filenames(instr_code, save_dir, split_freq)

    for file in file_names:
        instr_code = _instr_code_from_file_name(file, split_freq=split_freq)
        if split_freq:
            res = _res_from_file_name(file)
        else:
            res = None
        contract_date = _contract_date_from_file_name(file)
        contract_id = _get_barchart_id(
            instr_code, contract_date.year, contract_date.month
        )

        if contract_date > from_date:
            if dry_run:
                print(f"DRY RUN: would update contract {contract_id}, file {file}")
            else:
                try:
                    update_barchart_contract_file(
                        session, contract_map, save_dir, contract_id, res
                    )
                except IntegrityException:
                    logger.error(f"File index problem with {file}, please check")
                    check_integrity_list.append(file)
                except RecentUpdateException:
                    logger.warning(f"Skipping {contract_id}, recently updated")
                except EmptyDataException:
                    logger.info(f"Empty data for {contract_id}")

    if len(check_integrity_list) > 0:
        print(f"These files have integrity problems: {check_integrity_list}")


def _get_filenames(instr_code, save_dir, split_freq: bool = True):
    file_names = []
    if split_freq:
        for res in Resolution:
            regex = re.compile("^" + res.name + "_" + instr_code + "_[0-9]{8}.csv")
            file_names.extend([fn for fn in os.listdir(save_dir) if regex.match(fn)])
    else:
        regex = re.compile("^" + instr_code + "_[0-9]{8}.csv")
        file_names.extend([fn for fn in os.listdir(save_dir) if regex.match(fn)])

    return file_names


def update_barchart_contract_file(
    session: requests.Session,
    contract_map: dict,
    path: str,
    contract_id: str,
    res: Resolution,
):
    """
    Update a previously downloaded contract price file.

    Args:
        session: requests.Session instance
        contract_map: dict containing instrument config
        path: full path to the directory where previously downloaded files are located
        contract_id: Barchart style contract identifier, eg GCH24 for March 2024 Gold
        res: Resolution.Hour or Resolution.Day
    Raises:
        IntegrityException: raised if a problem is encountered when trying to set the
            datetime column as index
        RecentUpdateException: raised if the file has been recently updated
        EmptyDataException: raised if the update contains no data
    """
    inv_contract_map = _build_inverse_map(contract_map)

    file = _filename_from_barchart_id(contract_id, inv_contract_map, res)
    instr_code = _instr_code_from_file_name(file, res is not None)

    now = datetime.now().astimezone(tz=pytz.utc)

    input_path = f"{path}/{file}"
    logger.info(f"Starting update for {input_path}...")

    existing = pd.read_csv(input_path)
    existing["Time"] = pd.to_datetime(existing["Time"], format="%Y-%m-%dT%H:%M:%S%z")
    try:
        existing.set_index("Time", inplace=True, verify_integrity=True)
        last_index_date = existing.index[-1]
    except ValueError:
        raise IntegrityException(f"Index problem with {file}, needs manual check")

    if (now - last_index_date).days < 4:
        raise RecentUpdateException(f"Skipping {file}, recently updated")

    logger.info(
        f"Instrument: {instr_code}, contract: {contract_id}, "
        f"last entry: {last_index_date}"
    )

    update = get_historical_prices_for_contract(session, contract_id, res)
    if res == Resolution.Hour:
        start = last_index_date + timedelta(hours=1)
    else:
        start = last_index_date + timedelta(hours=25)
    end = now - timedelta(days=2)

    if update is not None:
        logger.info(
            f"Adding new rows from {start.strftime('%Y-%m-%d')} to "
            f"{end.strftime('%Y-%m-%d')}"
        )
        update = update[start:end]

        try:
            final = pd.concat([existing, update], verify_integrity=True)
            output_path = f"{path}/{file}"
            final.to_csv(output_path, date_format="%Y-%m-%dT%H:%M:%S%z")
        except Exception as ex:
            logger.warning(f"Problem with {file}: {ex}")
    else:
        raise EmptyDataException(f"Empty data for {contract_id}")


def get_historical_prices_for_contract(
    session, instr_code: str, resolution: Resolution = Resolution.Day
) -> pd.DataFrame:
    if not instr_code:
        raise BCException("instr_code is required")

    try:
        # GET the futures quote chart page, scrape to get XSRF token
        # https://www.barchart.com/futures/quotes/GCM21/interactive-chart
        chart_url = BARCHART_URL + f"futures/quotes/{instr_code}/interactive-chart"
        chart_resp = session.get(chart_url)
        xsrf = urllib.parse.unquote(chart_resp.cookies["XSRF-TOKEN"])

        headers = {
            "content-type": "text/plain; charset=UTF-8",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Referer": chart_url,
            "x-xsrf-token": xsrf,
        }

        payload = {
            "symbol": instr_code,
            "maxrecords": "640",
            "volume": "contract",
            "order": "asc",
            "dividends": "false",
            "backadjust": "false",
            "daystoexpiration": "1",
            "contractroll": "combined",
        }

        if resolution == Resolution.Day:
            data_url = BARCHART_URL + "proxies/timeseries/historical/queryeod.ashx"
            payload["data"] = "daily"
        else:
            data_url = BARCHART_URL + "proxies/timeseries/historical/queryminutes.ashx"
            payload["interval"] = "60"

        # get prices for instrument from BC internal API
        prices_resp = session.get(data_url, headers=headers, params=payload)
        if prices_resp.status_code != 200:
            raise Exception(
                f"response status: {prices_resp.status_code} {prices_resp.reason}"
            )
        ratelimit = prices_resp.headers["x-ratelimit-remaining"]
        if int(ratelimit) <= 15:
            time.sleep(20)
        logger.info(
            f"GET {data_url} {instr_code}, {prices_resp.status_code}, "
            f"ratelimit {ratelimit}"
        )

        # read response into dataframe
        iostr = io.StringIO(prices_resp.text)
        df = pd.read_csv(iostr, header=None)

        # convert to expected format
        price_data_as_df = _raw_barchart_data_to_df(df, bar_freq=resolution)

        if len(df) == 0:
            raise BCException(f"Zero length Barchart price data found for {instr_code}")

        logger.debug(f"Latest price {df.index[-1]} with {resolution}")

        return price_data_as_df

    except Exception as ex:
        logger.error(f"Problem getting historical data: {ex}")
        raise BCException from ex


def _build_contract_list(start_year, end_year, instr_list=None, contract_map=None):
    contracts_per_instrument = {}
    contract_list = []
    count = 0

    if contract_map is None:
        contract_map = CONTRACT_MAP

    if instr_list is None:
        instr_list = contract_map.keys()

    for instr in instr_list:
        config_obj = contract_map[instr]
        futures_code = config_obj["code"]
        if futures_code == "none":
            continue
        rollcycle = config_obj["cycle"]
        instrument_list = []

        for year in range(start_year, end_year):
            for month_code in list(rollcycle):
                instrument_list.append(
                    f"{futures_code}{month_code}{str(year)[len(str(year))-2:]}"
                )
        contracts_per_instrument[instr] = instrument_list
        logger.info(f"Adding {len(instrument_list)} contracts for {instr}")
        count = count + len(instrument_list)

    logger.info(f"Contract count: {count}")

    pool = cycle(contract_map.keys())

    # Count how many contracts are actually available to prevent infinite loops
    available_contracts = sum(
        len(contracts_per_instrument.get(instr, [])) for instr in contract_map.keys()
    )
    if available_contracts < count:
        logger.warning(
            f"Only {available_contracts} contracts available but count is set "
            f"to {count}. Adjusting count."
        )
        count = available_contracts

    while len(contract_list) < count:
        try:
            instr = next(pool)
        except StopIteration:
            logger.warning("Reached the end of the pool unexpectedly")
            continue
        if instr not in contracts_per_instrument:
            continue
        instr_list = contracts_per_instrument[instr]
        config_obj = contract_map[instr]
        rollcycle = config_obj["cycle"]
        if len(rollcycle) > 10:
            max_count = 3
        elif len(rollcycle) > 7:
            max_count = 2
        else:
            max_count = 1

        for _ in range(0, max_count):
            if len(instr_list) > 0:
                contract_list.append(instr_list.pop())

    # return ['CTH21', 'CTK21', 'CTN21', 'CTU21', 'CTZ21', 'CTH22']

    logger.info(f"Contract list: {contract_list}")
    return contract_list


def _build_inverse_map(contract_map):
    return {v["code"]: k for k, v in contract_map.items()}


def _before_available_res(resolution, start_date, instr_config):
    if "exchange" in instr_config:
        exch = instr_config["exchange"]
        if exch not in EXCHANGES:
            raise BCException(f"Missing exchange config for {exch}")
        exch_config = EXCHANGES[exch]
        tick_date = datetime.strptime(exch_config["tick_date"], "%Y-%m-%d")
        eod_date = datetime.strptime(exch_config["eod_date"], "%Y-%m-%d")

        if resolution == Resolution.Hour:
            return tick_date is not None and start_date < tick_date
        else:
            return eod_date is not None and start_date < eod_date
    else:
        raise BCException(f"No exchange specified for {instr_config['code']}")


def _get_overview(session, contract_id):
    """
    GET the futures overview page, e.g.
        https://www.barchart.com/futures/quotes/B6M21/overview
    :param contract_id: contract identifier
    :type contract_id: str
    :return: resp
    :rtype: HTTP response object
    """
    url = BARCHART_URL + "futures/quotes/%s/overview" % contract_id
    resp = session.get(url)
    logger.debug(f"GET {url}, response {resp.status_code}")
    return resp


def _build_save_path(instr_code, month, year, res: Resolution, save_directory):
    if save_directory is None:
        download_dir = os.getcwd()
    else:
        download_dir = save_directory
    datecode = str(year) + "{0:02d}".format(month)
    filename = f"{res.name}_{instr_code}_{datecode}00.csv"
    save_path = f"{download_dir}/{filename}"
    return save_path


def _get_contract_month_year(contract):
    year_code = int(contract[len(contract) - 2 :])
    month_code = contract[len(contract) - 3]
    if year_code > 30:
        year = 1900 + year_code
    else:
        year = 2000 + year_code
    month = _month_from_contract_letter(month_code.upper())
    return month, year


def _insufficient_data(session, symbol: str, res: Resolution):
    try:
        df = get_historical_prices_for_contract(session, symbol, res)
        return len(df) < 30
    except Exception:  # skipcq broad by design
        return True


def _get_start_end_dates(month, year, instr_config=None, default_day_count: int = 400):
    now = datetime.now()
    if instr_config and "days_count" in instr_config:
        day_count = instr_config["days_count"]
    else:
        day_count = default_day_count

    # we need to work out a date range for which we want the prices
    # for expired contracts the end date would be the expiry date;
    # for KISS sake, lets assume expiry is last date of contract month
    end_date = datetime(year, month, calendar.monthrange(year, month)[1])

    # but, if that end_date is in the future, then we may as well make it today...
    if now.date() < end_date.date():
        end_date = now

    # let's set start date at <day_count> days before end date
    day_count = timedelta(days=day_count)
    start_date = end_date - day_count

    return start_date, end_date


def _month_from_contract_letter(contract_letter):
    """
    Returns month number (1 is January) from contract letter

    :param contract_letter:
    :return:
    """
    try:
        month_number = MONTH_LIST.index(contract_letter)
    except ValueError:
        return None

    return month_number + 1


def _get_resolution(save_path):
    path_obj = Path(save_path)
    resol_str = path_obj.name.split("_")[0]
    try:
        return Resolution[resol_str]
    except KeyError:
        raise BCException(f"Unknown resolution: {resol_str}")


def _raw_barchart_data_to_df(
    price_data_raw: pd.DataFrame,
    bar_freq: Resolution = Resolution.Day,
) -> pd.DataFrame:
    if price_data_raw is None:
        logger.warning("No historical price data from Barchart")
        return pd.DataFrame([])

    if bar_freq == Resolution.Day:
        dateformat = "%Y-%m-%d"
        col_no = 1
        cols_to_remove = [0, 1, 6]
    else:
        dateformat = "%Y-%m-%d %H:%M"
        col_no = 0
        cols_to_remove = [0, 1]

    price_data_raw["Date"] = pd.to_datetime(price_data_raw[col_no], format=dateformat)
    price_data_raw.set_index("Date", inplace=True)
    price_data_raw.index = price_data_raw.index.tz_localize(tz="US/Central").tz_convert(
        "UTC"
    )
    price_data_raw.index.name = "Time"
    df = price_data_raw.drop(columns=cols_to_remove)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    return df


def _get_barchart_id(instr, year, month):
    instr_config = CONTRACT_MAP[instr]
    bc_instr = instr_config["code"]
    month_code = MONTH_LIST[month - 1]
    year_sub = year - 2000 if year > 2000 else year - 1900
    bc_id = f"{bc_instr}{month_code}{year_sub}"
    return bc_id


def _instr_code_from_file_name(file_name, split_freq: bool = True):
    if split_freq:
        instr_code = file_name[file_name.find("_") + 1 : file_name.rfind("_")]
    else:
        instr_code = file_name[: file_name.rfind("_")]
    return instr_code


def _res_from_file_name(file_name):
    res_str = file_name[: file_name.find("_")]
    res = Resolution[res_str]
    return res


def _contract_date_from_file_name(file_name):
    date_str = file_name[-12:-4]
    logger.debug(f"file: {file_name}, date_str: {date_str}")
    contract_date = datetime.strptime(f"{date_str[:-2]}01", "%Y%m01")
    return contract_date


def _filename_from_barchart_id(contract_id, inv_map, res: Resolution):
    try:
        month, year = _get_contract_month_year(contract_id)
        market_code = contract_id[: len(contract_id) - 3]
        instrument = inv_map[market_code.upper()]
        datecode = str(year) + "{0:02d}".format(month)
        if res is None:
            filename = f"{instrument}_{datecode}00.csv"
        else:
            filename = f"{res.name}_{instrument}_{datecode}00.csv"
        return filename
    except Exception as ex:
        raise Exception(f"Problem creating filename: {ex}")


def _env():
    credentials = {
        "barchart_username": "BARCHART_USERNAME",
        "barchart_password": "BARCHART_PASSWORD",
    }
    barchart_config = {
        k: os.environ.get(v) for k, v in credentials.items() if v in os.environ
    }
    return barchart_config


def _get_exchange_for_code(session, contract_code: str):
    """
    Get the exchange for the given Barchart code

    Scrapes the info page for the given contract to grab the exchange
    :param contract_code:
    :return: str
    """
    try:
        resp = _get_overview(session, contract_code)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find(name="div", attrs={"class": "commodity-profile"})
            label = table.find(name="div", string="Exchange")
            exchange_raw = label.next_sibling.next_sibling  # whitespace counts
            exchange = exchange_raw.text.strip()
            return exchange
        if resp.status_code == 404:
            print(f"Barchart page for {contract_code} not found")

    except Exception as e:
        print("Error: %s" % e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_barchart_downloads(
        create_bc_session(config_obj=_env()),
        contract_map={
            "AUD": {"code": "A6", "cycle": "HMUZ", "exchange": "CME"},
        },
        save_dir="barchart_data",
        start_year=2020,
        end_year=2022,
        dry_run=False,
    )

    # do_barchart_updates(
    #     create_bc_session(config_obj=_env()),
    #     contract_map={
    #         "AUD": {"code": "A6", "cycle": "HMUZ", "exchange": "CME"},
    #     },
    #     save_dir="/home/user/barchart_data",
    #     start_year=2020,
    #     end_year=2022,
    #     dry_run=False,
    # )
    
CONTRACT_MAP = {
    "AEX": {"code": "AE", "cycle": "FGHJKMNQUVXZ", "exchange": "EuronextInd"},
    "ALUMINIUM": {"code": "AL", "cycle": "FGHJKMNQUVXZ", "exchange": "COMEX"},
    "ASX": {"code": "AP", "cycle": "FGHJKMNQUVXZ", "exchange": "SFE"},
    "AUD": {"code": "A6", "cycle": "HMUZ", "exchange": "CME"},
    "AUDJPY": {"code": "UK", "cycle": "HMUZ", "exchange": "CME"},
    "BB3M": {"code": "BR", "cycle": "HMUZ", "exchange": "CME"},
    "BBCOMM": {"code": "AH", "cycle": "HMUZ", "exchange": "CBOT"},
    "BEL20": {"code": "BE", "cycle": "FGHJKMNQUVXZ", "exchange": "EuronextInd"},
    "BITCOIN": {"code": "BA", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "BOBL": {"code": "HR", "cycle": "HMUZ", "exchange": "EUREX"},
    "BONO": {"code": "IZ", "cycle": "HMUZ", "exchange": "EUREX"},
    "BOVESPA": {"code": "IF", "cycle": "GJMQVZ", "exchange": "CME"},
    "BRE": {"code": "L6", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "BRENT": {"code": "QA", "cycle": "FGHJKMNQUVXZ", "exchange": "NYMEX"},
    "BRENT_W": {"code": "CB", "cycle": "FGHJKMNQUVXZ", "exchange": "ICE/EU/Com"},
    "BRENT-LAST": {"code": "TBZ", "cycle": "FGHJKMNQUVXZ", "exchange": "NYMEX"},
    "BTP": {"code": "II", "cycle": "HMUZ", "exchange": "EUREX"},
    "BTP3": {"code": "IA", "cycle": "HMUZ", "exchange": "EUREX"},
    "BUND": {"code": "GG", "cycle": "HMUZ", "exchange": "EUREX"},
    "BUTTER": {"code": "BD", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "BUXL": {"code": "GX", "cycle": "HMUZ", "exchange": "EUREX"},
    "CAC": {"code": "MX", "cycle": "FGHJKMNQUVXZ", "exchange": "EuronextInd"},
    "CAD": {"code": "D6", "cycle": "HMUZ", "exchange": "CME"},
    "CADJPY": {"code": "UF", "cycle": "HMUZ", "exchange": "CME"},
    "CAD10": {"code": "CG", "cycle": "HMUZ", "exchange": "TMX"},
    "CANOLA": {"code": "RS", "cycle": "FHKNX", "exchange": "ICE/CA"},
    "CH10": {"code": "CF", "cycle": "HMUZ", "exchange": "EUREX"},
    "CHEESE": {"code": "BJ", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "CHF": {"code": "S6", "cycle": "HMUZ", "exchange": "CME"},
    "CHFJPY": {"code": "UP", "cycle": "HMUZ", "exchange": "CME"},
    "CLP": {"code": "N5", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "CNH": {"code": "ZX", "cycle": "FGHJKMNQUVXZ", "exchange": "ICE/SG"},
    "CNH-onshore": {"code": "ZP", "cycle": "FGHJKMNQUVXZ", "exchange": "ICE/SG"},
    "COCOA_LDN": {"code": "CA", "cycle": "HKNUZ", "exchange": "ICE/EU/Com"},
    "COCOA_NY": {"code": "CC", "cycle": "HKNUZ", "exchange": "ICE/US"},
    "COFFEE": {"code": "KC", "cycle": "HKNUZ", "exchange": "ICE/US"},
    "COPPER": {"code": "HG", "cycle": "FHJMNUVZ", "exchange": "COMEX"},
    "COPPER-micro": {"code": "QL", "cycle": "HKNUZ", "exchange": "COMEX"},
    "CORN": {"code": "ZC", "cycle": "HKNUZ", "exchange": "CBOT"},
    "COTTON": {"code": "KG", "cycle": "HKNVZ", "exchange": "NYMEX"},
    "COTTON2": {"code": "CT", "cycle": "HKNVZ", "exchange": "ICE/US"},
    "CRUDE_ICE": {"code": "WI", "cycle": "FGHJKMNQUVXZ", "exchange": "ICE/EU/Com"},
    "CRUDE_W": {"code": "CL", "cycle": "FGHJKMNQUVXZ", "exchange": "NYMEX"},
    "CZK": {"code": "WC", "cycle": "HMUZ", "exchange": "CME"},
    "DAX": {"code": "DY", "cycle": "HMUZ", "exchange": "EUREX"},
    "DIVDAX-DIVI": {"code": "AI", "cycle": "HMUZ", "exchange": "EUREX"},
    "DJSTX-SMALL": {"code": "PT", "cycle": "HMUZ", "exchange": "EUREX"},
    "DOW": {"code": "YM", "cycle": "HMUZ", "exchange": "CBOT"},
    "DX": {"code": "DX", "cycle": "HMUZ", "exchange": "ICE/US"},
    "ETHANOL": {"code": "ZK", "cycle": "FGHJKMNQUVXZ", "exchange": "CBOT"},
    "ETHEREUM": {"code": "ER", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "EUA": {
        "code": "CK",
        "cycle": "FGHJKMNQUVXZ",
        "days_count": 800,
        "exchange": "ENDEX",
    },
    "EUR": {"code": "E6", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "EURAUD": {"code": "UC", "cycle": "HMUZ", "exchange": "CME"},
    "EURCAD": {"code": "EP", "cycle": "HMUZ", "exchange": "ICE/US"},
    "EURCHF": {"code": "RF", "cycle": "HMUZ", "exchange": "CME"},
    "EURGBP": {"code": "RP", "cycle": "HMUZ", "exchange": "CME"},
    "EURIBOR": {"code": "TV", "cycle": "HMUZ", "exchange": "EUREX"},
    "EURIBOR-ICE": {
        "code": "IM",
        "cycle": "HMUZ",
        "days_count": 800,
        "exchange": "ICE",
    },
    "EUROSTX": {"code": "FX", "cycle": "HMUZ", "exchange": "EUREX"},
    "EUROSTX-LARGE": {"code": "PF", "cycle": "HMUZ", "exchange": "EUREX"},
    "EUROSTX-SMALL": {"code": "PI", "cycle": "HMUZ", "exchange": "EUREX"},
    "EUROSTX200-LARGE": {"code": "PO", "cycle": "HMUZ", "exchange": "EUREX"},
    "EURO600": {"code": "FY", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-AUTO": {"code": "S5", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-BASIC": {"code": "UJ", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-BANKS": {"code": "FA", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-CHEM": {"code": "C1", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-CONSTRUCTION": {"code": "C1", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-DIV50": {"code": "DO", "cycle": "MZ", "exchange": "EUREX"},
    "EU-DIV30": {"code": "AQ", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-DJ-OIL": {"code": "D9", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-DJ-TECH": {"code": "UO", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-DJ-TELECOM": {"code": "J5", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-DJ-UTIL": {"code": "IE", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-HEALTH": {"code": "YS", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-INSURE": {"code": "DML", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-MEDIA": {"code": "UQ", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-MID": {"code": "PG", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-REALESTATE": {"code": "M4", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-OIL": {"code": "UH", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-TECH": {"code": "UO", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-TRAVEL": {"code": "J3", "cycle": "HMUZ", "exchange": "EUREX"},
    "EU-UTILS": {"code": "J4", "cycle": "HMUZ", "exchange": "EUREX"},
    "FANG": {"code": "FG", "cycle": "HMUZ", "exchange": "ICE/US"},
    "FED": {"code": "ZQ", "cycle": "FGHJKMNQUVXZ", "exchange": "CBOT"},
    "FEEDCOW": {"code": "GF", "cycle": "FHJKQUVX", "exchange": "CME"},
    "FTSE100": {"code": "X", "cycle": "HMUZ", "exchange": "ICE"},
    "FTSE250": {"code": "Y", "cycle": "HMUZ", "exchange": "ICE"},
    "FTSECHINAA": {"code": "HN", "cycle": "FGHJKMNQUVXZ", "exchange": "SGX"},
    "FTSECHINAH": {"code": "HA", "cycle": "FGHJKMNQUVXZ", "exchange": "HKFE"},
    "FTSETAIWAN": {"code": "T1", "cycle": "FGHJKMNQUVXZ", "exchange": "SGX"},
    "GAS-LAST": {"code": "HH", "cycle": "FGHJKMNQUVXZ", "exchange": "NYMEX"},
    "GAS_NL": {"code": "TG", "cycle": "FGHJKMNQUVXZ", "exchange": "ENDEX"},
    "GAS_UK": {"code": "NF", "cycle": "FGHJKMNQUVXZ", "exchange": "ICE/EU/Com"},
    "GAS_US": {"code": "NG", "cycle": "FGHJKMNQUVXZ", "exchange": "NYMEX"},
    "GAS_US_mini": {"code": "QG", "cycle": "FGHJKMNQUVXZ", "exchange": "NYMEX"},
    "GASOIL": {"code": "LF", "cycle": "FGHJKMNQUVXZ", "exchange": "ICE/EU/Com"},
    "GASOILINE": {"code": "RB", "cycle": "FGHJKMNQUVXZ", "exchange": "NYMEX"},
    "GAS-PEN": {"code": "HP", "cycle": "FGHJKMNQUVXZ", "exchange": "NYMEX"},
    "GBP": {"code": "B6", "cycle": "HMUZ", "exchange": "CME"},
    "GBPCHF": {"code": "SS", "cycle": "HMUZ", "exchange": "ICE/US"},
    "GBPEUR": {"code": "RP", "cycle": "HMUZ", "exchange": "CME"},
    "GBPJPY": {"code": "UR", "cycle": "HMUZ", "exchange": "CME"},
    "GICS": {"code": "GD", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "GILT": {"code": "G", "cycle": "HMUZ", "exchange": "ICE"},
    "GOLD": {"code": "GC", "cycle": "GJMQVZ", "exchange": "COMEX"},
    "GOLD_micro": {"code": "GR", "cycle": "GJMQVZ", "exchange": "COMEX"},
    "HANG": {"code": "HS", "cycle": "FGHJKMNQUVXZ", "exchange": "HKFE"},
    "HANGENT_mini": {"code": "IX", "cycle": "FGHJKMNQUVXZ", "exchange": "HKFE"},
    "HANGTECH": {"code": "HI", "cycle": "FGHJKMNQUVXZ", "exchange": "HKFE"},
    "HEATOIL": {"code": "HO", "cycle": "FGHJKMNQUVXZ", "exchange": "NYMEX"},
    "HEATOIL-ICE": {"code": "LO", "cycle": "FGHJKMNQUVXZ", "exchange": "ICE/EU/Com"},
    "HOUSE-US": {"code": "I6", "cycle": "FKQX", "exchange": "CME"},
    "IBEX": {"code": "EF", "cycle": "HMUZ", "exchange": "MEFF"},
    "IRON": {"code": "C0", "cycle": "FGHJKMNQUVXZ", "exchange": "SGX"},
    "INR": {"code": "U-", "cycle": "FGHJKMNQUVXZ", "exchange": "SGX"},
    "IRS": {"code": "WY", "cycle": "HMUZ", "exchange": "CME"},
    "JGB": {"code": "JX", "cycle": "HMUZ", "exchange": "SGX"},
    "JGB-SGX-mini": {"code": "JX", "cycle": "HMUZ", "exchange": "SGX"},
    "JPY": {"code": "J6", "cycle": "HMUZ", "exchange": "CME"},
    "JP-REALESTATE": {"code": "V.", "cycle": "HMUZ", "exchange": "JPX"},
    "JSE40": {"code": "LX", "cycle": "HMUZ", "exchange": "SAFEX"},
    "KRWUSD": {"code": "K9", "cycle": "FGHJKMNQUVXZ", "exchange": "SGX"},
    "LEANHOG": {"code": "HE", "cycle": "GJKMNQVZ", "exchange": "CME"},
    "LIVECOW": {"code": "LE", "cycle": "GJMQVZ", "exchange": "CME"},
    "LUMBER-new": {"code": "LB", "cycle": "FHKNUX", "exchange": "CME"},
    # Replace MSCIASIA with EUREX symbol if available to match pysystemtrade
    "MSCIASIA": {"code": "L5", "cycle": "HMUZ", "exchange": "ICE/US"},
    "MSCISING": {"code": "SV", "cycle": "FGHJKMNQUVXZ", "exchange": "SGX"},
    "MID-DAX": {"code": "DM", "cycle": "HMUZ", "exchange": "EUREX"},
    "MILK": {"code": "DL", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "MILKDRY": {"code": "DF", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "MILKWET": {"code": "DK", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "MILLWHEAT": {"code": "ML", "cycle": "HKUZ", "exchange": "EuronextCom"},
    "MSCIWORLD": {"code": "AC", "cycle": "HMUZ", "exchange": "EUREX"},
    "MXP": {"code": "M6", "cycle": "HMUZ", "exchange": "CME"},
    "MUMMY": {"code": "S.", "cycle": "HMUZ", "exchange": "JPX"},
    "NASDAQ": {"code": "NQ", "cycle": "HMUZ", "exchange": "CME"},
    "NASDAQ_micro": {"code": "NM", "cycle": "HMUZ", "exchange": "CME"},
    "NIFTY": {"code": "NH", "cycle": "FGHJKMNQUVXZ", "exchange": "SGX"},
    "NIKKEI": {"code": "NY", "cycle": "HMUZ", "exchange": "CME"},
    "NIKKEI400": {"code": "OC", "cycle": "HMUZ", "exchange": "JPX"},
    "NOK": {"code": "NR", "cycle": "HMUZ", "exchange": "CME"},
    "NZD": {"code": "N6", "cycle": "HMUZ", "exchange": "CME"},
    "OAT": {"code": "FN", "cycle": "HMUZ", "exchange": "EUREX"},
    "OATIES": {"code": "ZO", "cycle": "HKNUZ", "exchange": "CBOT"},
    "OJ": {"code": "OJ", "cycle": "FHKNUX", "exchange": "ICE/US"},
    "OMX": {"code": "FP", "cycle": "HMUZ", "exchange": "EUREX"},
    "OMXS30": {"code": "OX", "cycle": "FGHJKMNQUVXZ", "exchange": "OMX"},
    "PALLAD": {"code": "PA", "cycle": "HMUZ", "exchange": "NYMEX"},
    "PLAT": {"code": "PL", "cycle": "FJNV", "exchange": "NYMEX"},
    "PLN": {"code": "WP", "cycle": "HMUZ", "exchange": "CME"},
    "R1000": {"code": "TX", "cycle": "HMUZ", "exchange": "CME"},
    "RAPESEED": {"code": "XR", "cycle": "GKQX", "exchange": "EuronextCom"},
    "REDWHEAT": {"code": "KE", "cycle": "HKNUZ", "exchange": "KCBT"},
    "RICE": {"code": "ZR", "cycle": "FHKNUX", "exchange": "CBOT"},
    "ROBUSTA": {"code": "RM", "cycle": "FHKNUX", "exchange": "ICE/EU/Com"},
    "RUBBER": {"code": "W2", "cycle": "FGHJKMNQUVXZ", "exchange": "SGX"},
    "RUR": {"code": "R6", "cycle": "HMUZ", "exchange": "CME"},
    "RUSSELL": {"code": "QR", "cycle": "HMUZ", "exchange": "CME"},
    "SEK": {"code": "SK", "cycle": "HMUZ", "exchange": "CME"},
    "SGD_mini": {"code": "R$", "cycle": "FGHJKMNQUVXZ", "exchange": "SGX"},
    "SGX": {"code": "VS", "cycle": "FGHJKMNQUVXZ", "exchange": "SGX"},
    "SHATZ": {"code": "HF", "cycle": "HMUZ", "exchange": "EUREX"},
    "SILVER": {"code": "SI", "cycle": "HKNUZ", "exchange": "COMEX"},
    "SMI": {"code": "SZ", "cycle": "HMUZ", "exchange": "EUREX"},
    "SMI-MID": {"code": "SH", "cycle": "HMUZ", "exchange": "EUREX"},
    "SOFR": {"code": "SQ", "cycle": "HMUZ", "days_count": 1100, "exchange": "CME"},
    "SONIA3": {"code": "MZ", "cycle": "HMUZ", "days_count": 1100, "exchange": "CME"},
    "SOYBEAN": {"code": "ZS", "cycle": "FHKNQUX", "exchange": "CBOT"},
    "SOYBEAN_mini": {"code": "XK", "cycle": "FHKNQUX", "exchange": "CBOT"},
    "SOYMEAL": {"code": "ZM", "cycle": "FHKNQUVZ", "exchange": "CBOT"},
    "SOYOIL": {"code": "ZL", "cycle": "FHKNQUVZ", "exchange": "CBOT"},
    "SP400": {"code": "EW", "cycle": "HMUZ", "exchange": "CME"},
    "SP500": {"code": "ES", "cycle": "HMUZ", "exchange": "CME"},
    "SP500_micro": {"code": "ET", "cycle": "HMUZ", "exchange": "CME"},
    "STEEL": {"code": "HV", "cycle": "FGHJKMNQUVXZ", "exchange": "COMEX"},
    "SUGAR": {"code": "SW", "cycle": "HKQVZ", "exchange": "ICE/EU/Com"},
    "SUGAR11": {"code": "SB", "cycle": "HKNV", "days_count": 200, "exchange": "ICE/US"},
    "SWISSLEAD": {"code": "ST", "cycle": "HMUZ", "exchange": "EUREX"},
    "TECDAX": {"code": "DZ", "cycle": "HMUZ", "exchange": "EUREX"},
    "TOPIX": {"code": "TS", "cycle": "HMUZ", "exchange": "JPX"},
    "US10": {"code": "ZN", "cycle": "HMUZ", "exchange": "CBOT"},
    "US10U": {"code": "TN", "cycle": "HMUZ", "exchange": "CBOT"},
    "US2": {"code": "ZT", "cycle": "HMUZ", "exchange": "CBOT"},
    "US30": {"code": "UD", "cycle": "HMUZ", "exchange": "CBOT"},
    "US30Y": {"code": "ZB", "cycle": "HMUZ", "exchange": "CBOT"},
    "US5": {"code": "ZF", "cycle": "HMUZ", "exchange": "CBOT"},
    "US-DISCRETE": {"code": "BI", "cycle": "HMUZ", "exchange": "CME"},
    "US-ENERGY": {"code": "BM", "cycle": "HMUZ", "exchange": "CME"},
    "US-FINANCE": {"code": "BN", "cycle": "HMUZ", "exchange": "CME"},
    "US-HEALTH": {"code": "BS", "cycle": "HMUZ", "exchange": "CME"},
    "US-INDUSTRY": {"code": "JG", "cycle": "HMUZ", "exchange": "CME"},
    "US-MATERIAL": {"code": "JA", "cycle": "HMUZ", "exchange": "CME"},
    "US-PROPERTY": {"code": "BK", "cycle": "HMUZ", "exchange": "CME"},
    "US-REALESTATE": {"code": "DH", "cycle": "HMUZ", "exchange": "CBOT"},
    "US-STAPLES": {"code": "BL", "cycle": "HMUZ", "exchange": "CME"},
    "US-TECH": {"code": "JB", "cycle": "HMUZ", "exchange": "CME"},
    "US-UTILS": {"code": "JE", "cycle": "HMUZ", "exchange": "CME"},
    "US20": {"code": "ZZ", "cycle": "HMUZ", "exchange": "CBOT"},
    "US3": {"code": "ZE", "cycle": "HMUZ", "exchange": "CBOT"},
    "V2X": {"code": "DV", "cycle": "FGHJKMNQUVXZ", "exchange": "EUREX"},
    "VIX": {"code": "VI", "cycle": "FGHJKMNQUVXZ", "exchange": "CFE"},
    "VNKI": {"code": "Y.", "cycle": "FGHJKMNQUVXZ", "exchange": "JPX"},
    "WHEAT": {"code": "ZW", "cycle": "HKNUZ", "exchange": "CBOT"},
    "WHEAT_ICE": {"code": "LW", "cycle": "FHKNUX", "exchange": "ICE/EU/Com"},
    "WHEY": {"code": "DG", "cycle": "FGHJKMNQUVXZ", "exchange": "CME"},
    "YENEUR": {"code": "RY", "cycle": "HMUZ", "exchange": "CME"},
    "ZAR": {"code": "T6", "cycle": "HMUZ", "exchange": "CME"},
}

# source: https://www.barchart.com/solutions/data/futures
EXCHANGES = {
    "CBOT": {"tick_date": "2008-05-04", "eod_date": "1978-03-03"},
    "CFE": {"tick_date": "2009-11-24", "eod_date": "2004-04-02"},
    "CME": {"tick_date": "2008-05-05", "eod_date": "1978-03-23"},
    "COMEX": {"tick_date": "2008-05-04", "eod_date": "1978-02-27"},
    "ENDEX": {"tick_date": "2014-01-01", "eod_date": "2014-01-01"},
    "EUREX": {"tick_date": "1999-06-23", "eod_date": "1990-11-28"},
    "EuronextCom": {"tick_date": "2011-05-25", "eod_date": "2001-09-17"},
    "EuronextInd": {"tick_date": "2011-05-25", "eod_date": "2003-06-23"},
    "HKFE": {"tick_date": "2008-05-05", "eod_date": "1986-05-06"},
    "ICE": {"tick_date": "2008-05-04", "eod_date": "1978-08-07"},
    "ICE/CA": {"tick_date": "2008-05-04", "eod_date": "1979-01-23"},
    "ICE/EU/Com": {"tick_date": "2008-05-05", "eod_date": "1989-07-24"},
    "ICE/EU/Fin": {"tick_date": "1999-06-21", "eod_date": "1982-11-04"},
    "ICE/SG": {"tick_date": "2010-03-09", "eod_date": "2010-03-09"},
    "ICE/US": {"tick_date": "2008-05-04", "eod_date": "1978-08-07"},
    "JPX": {"tick_date": "1992-01-06", "eod_date": "1992-01-06"},
    "KCBT": {"tick_date": "2008-07-14", "eod_date": "1979-03-26"},
    "MEFF": {"tick_date": "1992-09-29", "eod_date": "1992-09-29"},
    "NYMEX": {"tick_date": "2008-05-05", "eod_date": "1978-05-15"},
    "OMX": {"tick_date": "2012-01-27", "eod_date": "2012-01-27"},
    "SAFEX": {"tick_date": "2004-01-02", "eod_date": "2004-01-02"},
    "SFE": {"tick_date": "2019-11-03", "eod_date": "1980-01-02"},
    "SGX": {"tick_date": "2008-05-05", "eod_date": "1986-09-03"},
    "TMX": {"tick_date": "2010-04-05", "eod_date": "1993-04-22"},
}