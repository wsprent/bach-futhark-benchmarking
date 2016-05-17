--
-- Simple map into a scan with addition.
--
-- ==
-- compiled input @ data/simple_scan1_size_100000.input
-- output @ data/simple_scan1_size_100000.output
-- compiled input @ data/simple_scan1_size_1000000.input
-- output @ data/simple_scan1_size_1000000.output
-- compiled input @ data/simple_scan1_size_10000000.input
-- output @ data/simple_scan1_size_10000000.output
--
fun [int] main([int] a) =
    let b = map(+10, a) in
    scan(+, 0, b)
