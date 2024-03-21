
with raw_data as (
select formatDateTime(timestamp, '%Y-%m-%d') AS day, sku_id, sku, price, sum(qty) as qty
from default.demand_orders t1
join (select * from default.demand_orders_status
where status_id not in (2, 7, 8)) t2
on t1.order_id=t2.order_id
group by day, sku_id, sku, price
)

select t2.day as day, t1.sku_id as sku_id, t1.sku as sku, t1.price as price, coalesce(raw_data.qty, 0) as qty
from 
(select distinct sku_id, sku, price from default.demand_orders) t1
cross join (select distinct formatDateTime(timestamp, '%Y-%m-%d') AS day from default.demand_orders) t2
left join raw_data on t1.sku_id=raw_data.sku_id and t2.day=raw_data.day
order by sku_id, day 

