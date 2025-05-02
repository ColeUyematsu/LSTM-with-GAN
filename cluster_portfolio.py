import random

def clean_data(data_unformatted):
    lines = data_unformatted.strip().split('\n')

    headers = [h.strip() for h in lines[0].split(',')][1:]

    data_dicts = []
    for line in lines[1:]:
        values = [float(v.strip()) for v in line.split(',')[1:]]
        row_dict = dict(zip(headers, values))
        data_dicts.append(row_dict)

    return data_dicts

def kmeans(data_dicts, k=3, max_iter=100, converge_threshold=1e-4):
    portfolio_names = list(data_dicts[0].keys())
    num_months = len(data_dicts)
    
    # Initialize centroids with random portfolios
    centroids = []
    for idx in random.sample(range(len(portfolio_names)), k):
        returns = [data_dicts[month][portfolio_names[idx]] for month in range(num_months)]
        centroids.append(returns)
    
    for _ in range(max_iter):
        # Assign clusters
        clusters = [[] for _ in range(k)]
        for port_idx, name in enumerate(portfolio_names):
            point = [data_dicts[month][name] for month in range(num_months)]
            distances = [sum((x-y)**2 for x,y in zip(point, centroid)) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(port_idx)
        
        # Update centroids
        new_centroids = []
        for cluster_idx, cluster in enumerate(clusters):
            if not cluster:  # Handle empty clusters
                new_centroids.append(centroids[cluster_idx])
                continue
            
            # Calculate mean for each month
            new_centroid = []
            for month in range(num_months):
                month_values = [data_dicts[month][portfolio_names[i]] for i in cluster]
                new_centroid.append(sum(month_values)/len(month_values))
            new_centroids.append(new_centroid)
        
        # Check for convergence
        converged = True
        for old, new in zip(centroids, new_centroids):
            if any(abs(x-y) > converge_threshold for x,y in zip(old, new)):
                converged = False
                break
        if converged:
            break
        centroids = new_centroids
    
    # Create final assignments
    assignments = [0] * len(portfolio_names)
    for cluster_idx, cluster in enumerate(clusters):
        for port_idx in cluster:
            assignments[port_idx] = cluster_idx
    
    return assignments, centroids

def run_kmeans():
    data = """Date, SMALL LoBM,ME1 BM2,ME1 BM3,ME1 BM4,SMALL HiBM,ME2 BM1,ME2 BM2,ME2 BM3,ME2 BM4,ME2 BM5,ME3 BM1,ME3 BM2,ME3 BM3,ME3 BM4,ME3 BM5,ME4 BM1,ME4 BM2,ME4 BM3,ME4 BM4,ME4 BM5,BIG LoBM,ME5 BM2,ME5 BM3,ME5 BM4,BIG HiBM
    202009,-1.2699,-0.3631,-4.5372,-3.6009,-2.9851,0.3682,-4.9898,-5.1021,-6.9521,-5.1615,-3.2829,-1.3979,-4.4291,-3.3158,-5.8876,1.0141,-2.6447,-3.5014,-3.0425,-4.8766,-4.5508,-3.5645,-1.2523,-4.6709,-6.1118
    202010,-2.684,-1.2718,-0.1601,3.0812,0.7294,-2.2818,4.0725,2.3729,5.6544,5.9573,2.9478,2.1655,2.8126,5.8353,5.919,1.3698,2.0723,1.8801,1.889,2.0096,-4.5711,0.003,-0.0869,-2.3244,0.9883
    202011,25.4603,21.6852,19.8299,20.5356,23.9605,21.3881,19.0704,17.9596,19.0574,19.3157,19.8737,16.9854,15.3246,16.6395,21.5046,13.8204,12.8281,15.9609,15.7106,20.4074,10.7257,9.8547,14.4605,16.1654,21.978
    202012,13.1824,7.8131,9.2485,8.438,7.1857,11.2639,10.0651,8.8478,8.5,6.9177,12.2766,8.6516,6.9268,9.5035,7.9595,6.8499,5.9012,4.7273,7.8074,5.438,5.3004,1.8685,3.418,3.2349,8.0465
    202101,16.6704,14.9932,8.1702,7.3884,39.7565,12.1049,5.7885,7.6529,3.723,8.9216,2.5181,0.9032,2.4192,1.8087,4.3874,-0.2722,0.6493,-0.215,2.3776,1.6662,-0.9357,-0.5953,-2.0951,0.8452,1.0989"""

    cleaned_data = clean_data(data)

    clusters = kmeans(cleaned_data)

    print(clusters)

if __name__ == '__main__':
    run_kmeans()