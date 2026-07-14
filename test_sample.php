<?php

class FinancialReportService
{
    private $db;
    private $logger;

    public function __construct($dbConnection, $logger)
    {
        $this->db = $dbConnection;
        $this->logger = $logger;
    }

    /**
     * Processes and generates the monthly financial report for a specific region.
     *
     * @param int $year The reporting year.
     * @param int $month The reporting month.
     * @param string $regionCode The region to generate the report for.
     * @return array Contains 'success', 'data', or 'error' message.
     */
    public function processMonthlyFinancialReport(int $year, int $month, string $regionCode): array
    {
        $this->logger->info("Starting financial report generation for $regionCode ($year-$month)");
        $reportData = [
            'metadata' => [
                'year' => $year,
                'month' => $month,
                'region' => $regionCode,
                'generated_at' => date('Y-m-d H:i:s')
            ],
            'summary' => [
                'total_revenue' => 0.0,
                'total_expenses' => 0.0,
                'net_profit' => 0.0,
                'transaction_count' => 0
            ],
            'details' => []
        ];

        try {
            // 1. Validate inputs
            if ($month < 1 || $month > 12) {
                throw new InvalidArgumentException("Invalid month provided: $month");
            }
            if (empty($regionCode)) {
                throw new InvalidArgumentException("Region code cannot be empty");
            }

            // 2. Fetch raw transactions for the month and region
            // (Mocking a complex DB query)
            $startDate = "$year-$month-01 00:00:00";
            $endDate = date("Y-m-t 23:59:59", strtotime($startDate));

            $query = "SELECT id, type, amount, category, created_at, status 
                      FROM transactions 
                      WHERE region_code = :region 
                      AND created_at BETWEEN :start AND :end 
                      AND status = 'COMPLETED'";

            $stmt = $this->db->prepare($query);
            $stmt->execute(['region' => $regionCode, 'start' => $startDate, 'end' => $endDate]);
            $transactions = $stmt->fetchAll(PDO::FETCH_ASSOC);

            if (empty($transactions)) {
                $this->logger->warning("No transactions found for $regionCode in $year-$month");
                return ['success' => true, 'data' => $reportData, 'message' => 'No data found.'];
            }

            // 3. Process and categorize transactions
            $categorizedTotals = [];
            foreach ($transactions as $txn) {
                $amount = (float) $txn['amount'];
                $cat = $txn['category'];

                // Track by category
                if (!isset($categorizedTotals[$cat])) {
                    $categorizedTotals[$cat] = ['revenue' => 0, 'expense' => 0];
                }

                if ($txn['type'] === 'CREDIT') {
                    $reportData['summary']['total_revenue'] += $amount;
                    $categorizedTotals[$cat]['revenue'] += $amount;
                } elseif ($txn['type'] === 'DEBIT') {
                    $reportData['summary']['total_expenses'] += $amount;
                    $categorizedTotals[$cat]['expense'] += $amount;
                }

                $reportData['summary']['transaction_count']++;
            }

            // 4. Calculate Net Profit and Taxes (Complex Business Logic)
            $reportData['summary']['net_profit'] = $reportData['summary']['total_revenue'] - $reportData['summary']['total_expenses'];

            $taxRate = 0.0;
            if ($regionCode === 'NA' && $reportData['summary']['net_profit'] > 50000) {
                $taxRate = 0.21; // 21% Corporate Tax
            } elseif ($regionCode === 'EU') {
                $taxRate = 0.19; // 19% standard rate
            }

            $estimatedTax = max(0, $reportData['summary']['net_profit'] * $taxRate);
            $reportData['summary']['estimated_tax'] = $estimatedTax;
            $reportData['summary']['net_profit_after_tax'] = $reportData['summary']['net_profit'] - $estimatedTax;

            // 5. Format detailed breakdown
            foreach ($categorizedTotals as $category => $totals) {
                $reportData['details'][] = [
                    'category_name' => strtoupper($category),
                    'total_revenue' => number_format($totals['revenue'], 2, '.', ''),
                    'total_expense' => number_format($totals['expense'], 2, '.', ''),
                    'margin' => $totals['revenue'] > 0
                        ? round((($totals['revenue'] - $totals['expense']) / $totals['revenue']) * 100, 2) . '%'
                        : 'N/A'
                ];
            }

            // 6. Save report to database (archive)
            $archiveQuery = "INSERT INTO monthly_reports (region, year, month, report_json, created_at) 
                             VALUES (:reg, :yr, :mo, :json, NOW())";
            $archiveStmt = $this->db->prepare($archiveQuery);
            $archiveStmt->execute([
                'reg' => $regionCode,
                'yr' => $year,
                'mo' => $month,
                'json' => json_encode($reportData)
            ]);

            $this->logger->info("Successfully generated report ID: " . $this->db->lastInsertId());

            return [
                'success' => true,
                'data' => $reportData
            ];

        } catch (PDOException $e) {
            $this->logger->error("Database error while generating report: " . $e->getMessage());
            return ['success' => false, 'error' => 'A database error occurred during generation.'];
        } catch (Exception $e) {
            $this->logger->error("Unexpected error generating report: " . $e->getMessage());
            return ['success' => false, 'error' => $e->getMessage()];
        }
    }
}
?>