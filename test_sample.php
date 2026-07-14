<?php
// test_sample.php
// Sample PHP file to test language detection

function calculateDiscount($price, $discountPercent)
{
    if ($price <= 0) {
        return 0;
    }
    $discount = ($discountPercent / 100) * $price;
    return $price - $discount;
}

function getUserInfo($userId, $users)
{
    foreach ($users as $user) {
        if ($user['id'] === $userId) {
            return $user;
        }
    }
    return null;
}

function formatReport($data)
{
    $lines = [];
    foreach ($data as $idx => $item) {
        $lines[] = ($idx + 1) . ". " . $item['name'] . " — Score: " . $item['score'];
    }
    return implode("\n", $lines);
}

$price = calculateDiscount(100, 20);
echo $price . "\n";

$user = getUserInfo(1, [["id" => 1, "name" => "Alice"]]);
echo $user['name'] . "\n";
?>