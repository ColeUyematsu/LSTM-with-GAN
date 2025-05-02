import random

def clean_data(data_unformatted):
    lines = data_unformatted.strip().split('\n')
    headers = [h.strip() for h in lines[0].split(',')][1:]
    data_dicts = []
    for line in lines[1:]:
        values = [float(v.strip()) for v in line.split(',')[1:]]
        row_dict = dict(zip(headers, values))
        data_dicts.append(row_dict)
    return data_dicts, headers

def kmeans(data_dicts, portfolio_names, k=3, max_iter=100, converge_threshold=1e-6):
    num_months = len(data_dicts)
    num_portfolios = len(portfolio_names)

    # initialize centroids by randomly picking three points
    centroids = []
    initial_centroid_indices = random.sample(range(num_portfolios), k)
    for idx in initial_centroid_indices:
        portfolio_name = portfolio_names[idx]
        returns = [data_dicts[month][portfolio_name] for month in range(num_months)]
        centroids.append(returns)

    assignments = [0] * num_portfolios

    for _ in range(max_iter):
        # assign clusters based on nearest centroid
        clusters = [[] for _ in range(k)]
        for port_idx, name in enumerate(portfolio_names):
            point = [data_dicts[month][name] for month in range(num_months)]
            distances = [sum((x-y)**2 for x,y in zip(point, centroid)) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(port_idx)
            assignments[port_idx] = cluster_idx 

        # update centroids by averaging portfolios in each cluster
        updated_centroids = []
        centroid_movement = []
        for cluster_idx, cluster_portfolio_indices in enumerate(clusters):
            if not cluster_portfolio_indices:
                updated_centroids.append(centroids[cluster_idx])
                centroid_movement.append((centroids[cluster_idx], centroids[cluster_idx]))
                continue

            new_centroid = [0.0] * num_months
            num_portfolios_in_cluster = len(cluster_portfolio_indices)
            for port_idx in cluster_portfolio_indices:
                portfolio_name = portfolio_names[port_idx]
                portfolio_series = [data_dicts[month][portfolio_name] for month in range(num_months)]
                for month in range(num_months):
                    new_centroid[month] += portfolio_series[month]
            for month in range(num_months):
                new_centroid[month] /= num_portfolios_in_cluster
            updated_centroids.append(new_centroid)
            centroid_movement.append((centroids[cluster_idx], new_centroid))

        # check for convergence
        converged = True
        for old_centroid, new_centroid in centroid_movement:
            if any(abs(x-y) > converge_threshold for x,y in zip(old_centroid, new_centroid)):
                converged = False
                break
        
        centroids = updated_centroids # update centroids
        
        if converged:
            break

    return assignments, centroids

def run_kmeans():
    data = """Date, SMALL LoBM,ME1 BM2,ME1 BM3,ME1 BM4,SMALL HiBM,ME2 BM1,ME2 BM2,ME2 BM3,ME2 BM4,ME2 BM5,ME3 BM1,ME3 BM2,ME3 BM3,ME3 BM4,ME3 BM5,ME4 BM1,ME4 BM2,ME4 BM3,ME4 BM4,ME4 BM5,BIG LoBM,ME5 BM2,ME5 BM3,ME5 BM4,BIG HiBM
    202009,-1.2699,-0.3631,-4.5372,-3.6009,-2.9851,0.3682,-4.9898,-5.1021,-6.9521,-5.1615,-3.2829,-1.3979,-4.4291,-3.3158,-5.8876,1.0141,-2.6447,-3.5014,-3.0425,-4.8766,-4.5508,-3.5645,-1.2523,-4.6709,-6.1118
    202010,-2.684,-1.2718,-0.1601,3.0812,0.7294,-2.2818,4.0725,2.3729,5.6544,5.9573,2.9478,2.1655,2.8126,5.8353,5.919,1.3698,2.0723,1.8801,1.889,2.0096,-4.5711,0.003,-0.0869,-2.3244,0.9883
    202011,25.4603,21.6852,19.8299,20.5356,23.9605,21.3881,19.0704,17.9596,19.0574,19.3157,19.8737,16.9854,15.3246,16.6395,21.5046,13.8204,12.8281,15.9609,15.7106,20.4074,10.7257,9.8547,14.4605,16.1654,21.978
    202012,13.1824,7.8131,9.2485,8.438,7.1857,11.2639,10.0651,8.8478,8.5,6.9177,12.2766,8.6516,6.9268,9.5035,7.9595,6.8499,5.9012,4.7273,7.8074,5.438,5.3004,1.8685,3.418,3.2349,8.0465
    202101,16.6704,14.9932,8.1702,7.3884,39.7565,12.1049,5.7885,7.6529,3.723,8.9216,2.5181,0.9032,2.4192,1.8087,4.3874,-0.2722,0.6493,-0.215,2.3776,1.6662,-0.9357,-0.5953,-2.0951,0.8452,1.0989"""

    cleaned_data, portfolio_names = clean_data(data)

    random.seed(45) # use random seed for repeatable results

    cluster_assignments = kmeans(cleaned_data, portfolio_names)[0]

    print("CLUSTERS:")
    for cluster_num in range(3):
        print(f"\nCluster {cluster_num + 1}:")
        cluster_portfolios = [portfolio_names[i] for i, assignment in enumerate(cluster_assignments) if assignment == cluster_num]
        print(", ".join(cluster_portfolios))

if __name__ == '__main__':
    run_kmeans()