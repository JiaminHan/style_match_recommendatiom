# shopstyle
Final Project at Metis
Shopping for fashionable yet affordable outfits is a never ending quest. Trendy outfit styles are generally first invented by designer brands, whose products are too expensive for average consumers. The high demand for cheaper alternatives to designer outfits has driven many consumer fashion brands, such as H&M and ZARA, to make knockoff styles. However, it is still difficult for shoppers to find a cheap alternative to a specific designer outfit, because shoppers have to first find the products that match the target style across different websites, and then find the cheapest from the matched products. The problem is even more challenging if the shopper wants to replicate the entire outfit. It is therefore desirable to develop a deep learning algorithm to accomplish this task. The goal of my project is to find more affordable alternatives to designer outfit by using convolutional neural networks and deep learning techniques. 

My project aims to build a recommendation system based on a single input image. The system is comprised of two main parts:
Segment and localize the objects of interest in a target designer outfit image
Retrieve similar and cheaper products for each object in the target outfit from retailer’s database.

I will collect images and products information from shopstyle.com using its API. By using Convolutional Neural Network (CNN), I want to first extract features from images of products by different clothes categories(e.g. Dresses, blouses, pants, shoes). Then I will cluster the features so that I can divide the products into several unique groups. Within each group, I will sort the products by similarity and price, so that I can find the overall similar but more affordable outfits combinations as recommendations.
