import pandas as pd
from vnstock3 import Vnstock
from datetime import date
from typing import List, Optional

def get_data(tickers: Optional[List[str]] = None, 
             start: str = '2018-01-01',
             interval: str = '1D') -> pd.DataFrame:
    """
    Lấy dữ liệu thị trường chứng khoán (OHLCVS) từ ngày 'start' đến ngày hiện tại cho các mã cổ phiếu.
    
    Hàm này cho phép bạn lấy dữ liệu cho:
      - Thị trường Việt Nam (VN) sử dụng module Vnstock.
      
    Parameters
    ----------
    tickers : Optional[List[str]], default=None
        Danh sách các mã cổ phiếu cần lấy dữ liệu. Nếu không cung cấp, hàm sẽ sử dụng danh sách mã mặc định tùy theo thị trường.
    start : str, default='2018-01-01'
        Ngày bắt đầu lấy dữ liệu theo định dạng 'YYYY-MM-DD'.
    interval : str, default='1D
        Khoảng cách cây thời gian của dữ liệu
    
    Returns
    -------
    pd.DataFrame
        DataFrame chứa dữ liệu ở định dạng OHLCVS với các cột:
          - open, high, low, close, volume, symbol
        Các dữ liệu được sắp xếp theo ngày tăng dần.
    
    Notes
    -----
    - Đối với thị trường VN, dữ liệu được lấy qua module Vnstock với nguồn 'VCI'. 
      Giá trị các cột giá (open, high, low, close) được nhân với 1000 và ép kiểu thành int.
    """
    # Lấy ngày hiện tại
    today = str(date.today())
    
    # Đặt các mã cổ phiếu muốn chọn
    if tickers is not None:
        symbols = tickers
    else:
        symbols = [
            "ACB", "BID", "BVH", "CTG", "FPT", "GAS",
            "GVR", "HDB", "HPG", "KDH", "MBB", "MSN",
            "MWG", "NVL", "PDR", "PLX", "PNJ", "POW",
            "SAB", "SSI", "STB", "TCB", "TPB", "VCB",
            "VHM", "VIC", "VJC", "VNM", "VPB", "VRE"
        ]
            
    # Lấy dữ liệu của thị trường Việt Nam
    stock = Vnstock().stock(source='VCI')
    df = stock.quote.history(symbol=symbols[0], start=start, end=today, interval=interval)
    df['symbol'] = symbols[0]
    for i in range(1, len(symbols)):
        b = stock.quote.history(symbol=symbols[i], start=start, end=today, interval=interval)
        b['symbol'] = symbols[i]
        df = pd.concat([df, b], axis=0)
    df.set_index('time', inplace=True)
    df.index.name = 'date'
    df.sort_index(inplace=True)
    if interval != '1D':
        df = fill_missing_weeks_multiple_stocks(df)
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']] * 1000
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(int)
            
    return df.sort_index()

def fill_missing_weeks_multiple_stocks(df):
    # Đảm bảo index là datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Tạo danh sách để lưu dữ liệu đã xử lý
    processed_data = []
    
    # Xử lý cho từng mã chứng khoán
    for symbol in df['symbol'].unique():
        # Lọc dữ liệu cho mã chứng khoán này
        symbol_data = df[df['symbol'] == symbol].copy()
        
        # Tạo một index hoàn chỉnh với tất cả các tuần
        start_date = symbol_data.index.min()
        end_date = symbol_data.index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='W')
        
        # Tạo DataFrame mới với đầy đủ các tuần
        reindexed_data = symbol_data.reindex(full_date_range)
        
        # Điền lại symbol vì có thể bị mất sau reindex
        reindexed_data['symbol'] = symbol
        
        # Forward fill cho các giá trị OHLC
        reindexed_data[['open', 'high', 'low', 'close', 'volume']] = reindexed_data[['open', 'high', 'low', 'close', 'volume']].fillna(method='ffill')
        
        # Thêm vào danh sách kết quả
        processed_data.append(reindexed_data)
    
    # Ghép tất cả dữ liệu đã xử lý
    result_df = pd.concat(processed_data)
    
    return result_df

def transform(df: pd.DataFrame,
              component: str = 'close',
              fill: bool = False) -> pd.DataFrame:
    """
    Chuyển đổi dữ liệu OHLCVS thành dạng chứa thành phần giá được chọn duy nhất hoặc giữ lại toàn bộ dữ liệu (multi-index).

    Hàm này thực hiện các bước sau:
      - Sao chép DataFrame gốc để không làm thay đổi dữ liệu ban đầu.
      - Nếu tham số `component` khác 'all', tạo bảng pivot với:
            - index: ngày (date)
            - columns: mã cổ phiếu (symbol)
            - values: giá trị của thành phần được chỉ định (ví dụ: open, high, low, close)
      - Nếu `component` là 'all', tạo bảng pivot giữ lại tất cả các cột OHLCVS cho mỗi symbol.
      - Nếu `fill` là True, áp dụng forward fill và backward fill cho dữ liệu bị thiếu. Nếu thị trường là 'VN',
        các giá trị sẽ được ép kiểu thành số nguyên.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu ở định dạng OHLCVS, bao gồm các cột như 'open', 'high', 'low', 'close', 'volume', và 'symbol'.
    component : str, default='close'
        Thành phần giá cần giữ lại. Có thể là 'open', 'high', 'low', 'close' hoặc 'all'. 
        Nếu là 'all', giữ lại toàn bộ dữ liệu (tạo thành multi-index).
    fill : bool, default=False
        Nếu True, áp dụng forward fill và backward fill cho dữ liệu bị thiếu.

    Returns
    -------
    pd.DataFrame
        DataFrame chứa dữ liệu đã được chuyển đổi:
          - Nếu `component` khác 'all', các cột là các mã cổ phiếu (symbol) với giá trị tương ứng của thành phần đã chọn.
          - Nếu `component` là 'all', DataFrame giữ lại toàn bộ dữ liệu theo cấu trúc multi-index.
    """
    data = df.copy()
    if component != 'all':
        pivot = data.pivot_table(index=data.index, columns='symbol', values=component)
    else:
        pivot = data.pivot_table(index=data.index, columns='symbol')
        
    if fill:
        pivot.ffill(inplace=True)
        pivot.bfill(inplace=True)
        pivot = pivot.astype(int)
        
    return pivot