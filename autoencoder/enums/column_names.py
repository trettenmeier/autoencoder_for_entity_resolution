from enum import Enum


class IncomingColName(Enum):
    client_oem_info_brand = 'client_oem_info_brand'
    client_oem_info_number = 'client_oem_info_number'
    client_oem_info_part_type = 'client_oem_info_part_type'
    client_sales_info_price_sell = 'client_sales_info_price_sell'
    client_internal_info_group = 'client_internal_info_group'

    page_oem_info_brand = 'page_oem_info_brand'
    page_oem_info_number = 'page_oem_info_number'
    page_oem_info_part_type = 'page_oem_info_part_type'
    page_internal_info_name = 'page_internal_info_name'
    page_price = 'page_price'
    page_internal_info_description = 'page_internal_info_description'
    page_internal_info_group = 'page_internal_info_group'
