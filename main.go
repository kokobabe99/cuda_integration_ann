package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"time"
)

/***************
 *  Embedding  *
 ***************/

const (
	dim        = 64
	alpha      = 0.7
	randomSeed = 2025
)

type vec = []float32

var numberBase [76]vec
var posBase [25]vec

func initBases() {
	rng := rand.New(rand.NewSource(randomSeed))
	gen := func() vec {
		v := make(vec, dim)
		for i := range v {
			v[i] = float32(rng.Float64()*2 - 1) // [-1,1]
		}
		normInPlace(v)
		return v
	}
	for n := 1; n <= 75; n++ {
		numberBase[n] = gen()
	}
	for i := 0; i < 25; i++ {
		posBase[i] = gen()
	}
}

func normInPlace(v vec) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	n := float32(math.Sqrt(s) + 1e-12)
	for i := range v {
		v[i] /= n
	}
}

func cardToVec(card [25]int) vec {
	out := make(vec, dim)
	for i, n := range card {
		if n < 1 || n > 75 {
			panic("Bingo number out of range [1..75]")
		}
		b := numberBase[n]
		p := posBase[i]
		for j := 0; j < dim; j++ {
			out[j] += b[j] + float32(alpha)*p[j]
		}
	}
	normInPlace(out)
	return out
}

func cosineDist(a, b vec) float32 {
	var dot float64
	for i := 0; i < dim; i++ {
		dot += float64(a[i]) * float64(b[i])
	}
	return float32(1 - dot)
}

/**********************
 *  K-Means ANN Index *
 **********************/

type KMeansANN struct {
	k         int
	maxIters  int
	centroids []vec
	lists     [][]int
	data      []vec
	rng       *rand.Rand
}

// 构建：输入全量向量，训练 k-means 并建立倒排表
func NewKMeansANN(all []vec, k, maxIters int) *KMeansANN {
	if k <= 1 {
		k = 2
	}
	if maxIters <= 0 {
		maxIters = 20
	}
	ann := &KMeansANN{
		k:        k,
		maxIters: maxIters,
		data:     all,
		rng:      rand.New(rand.NewSource(randomSeed + 202)),
	}
	centroids, assign := ann.trainKMeansPlusPlus(all, k, maxIters)
	ann.centroids = centroids
	ann.buildInvertedLists(assign, len(all))
	return ann
}

func (a *KMeansANN) trainKMeansPlusPlus(all []vec, k, maxIters int) ([]vec, []int) {
	n := len(all)
	if n == 0 {
		log.Fatal("empty dataset")
	}
	if k > n {
		k = n
	}

	centroids := make([]vec, 0, k)

	first := a.rng.Intn(n)
	centroids = append(centroids, append(vec(nil), all[first]...))

	dist2 := make([]float64, n)
	for c := 1; c < k; c++ {
		var sum float64
		for i := 0; i < n; i++ {
			// 到已选中心的最小距离
			best := math.MaxFloat64
			for _, ctr := range centroids {
				d := float64(cosineDist(all[i], ctr))
				if d < best {
					best = d
				}
			}
			dist2[i] = best * best
			sum += dist2[i]
		}
		if sum == 0 {
			// 数据都重合，复制一个中心
			centroids = append(centroids, append(vec(nil), all[a.rng.Intn(n)]...))
			continue
		}
		r := a.rng.Float64() * sum
		acc := 0.0
		chosen := 0
		for i := 0; i < n; i++ {
			acc += dist2[i]
			if acc >= r {
				chosen = i
				break
			}
		}
		centroids = append(centroids, append(vec(nil), all[chosen]...))
	}

	// 3) Lloyd 迭代
	assign := make([]int, n)
	for it := 0; it < maxIters; it++ {
		// E-step: 重新分配
		changed := 0
		for i := 0; i < n; i++ {
			bestC := 0
			bestD := cosineDist(all[i], centroids[0])
			for c := 1; c < k; c++ {
				d := cosineDist(all[i], centroids[c])
				if d < bestD {
					bestD = d
					bestC = c
				}
			}
			if assign[i] != bestC || it == 0 {
				if assign[i] != bestC {
					changed++
				}
				assign[i] = bestC
			}
		}

		sums := make([]vec, k)
		counts := make([]int, k)
		for c := 0; c < k; c++ {
			sums[c] = make(vec, dim)
		}
		for i := 0; i < n; i++ {
			c := assign[i]
			counts[c]++
			v := all[i]
			for j := 0; j < dim; j++ {
				sums[c][j] += v[j]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				// 空簇：随机重置为某个点
				idx := a.rng.Intn(n)
				copy(centroids[c], all[idx])
				continue
			}
			for j := 0; j < dim; j++ {
				sums[c][j] /= float32(counts[c])
			}
			normInPlace(sums[c]) // 归一化中心，配合 cosine
			centroids[c] = sums[c]
		}

		if changed == 0 && it > 0 {
			break
		}
	}
	return centroids, assign
}

func (a *KMeansANN) buildInvertedLists(assign []int, n int) {
	lists := make([][]int, a.k)
	for i := 0; i < a.k; i++ {
		lists[i] = make([]int, 0)
	}
	for i := 0; i < n; i++ {
		c := assign[i]
		lists[c] = append(lists[c], i)
	}
	a.lists = lists
}

func (a *KMeansANN) Search(q vec, topK, nprobe int) ([]int, []float32) {
	if nprobe <= 0 {
		nprobe = 1
	}
	if nprobe > a.k {
		nprobe = a.k
	}
	type cd struct {
		c    int
		dist float32
	}
	cs := make([]cd, a.k)
	for c := 0; c < a.k; c++ {
		cs[c] = cd{c: c, dist: cosineDist(q, a.centroids[c])}
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].dist < cs[j].dist })
	cs = cs[:nprobe]

	// 2) 收集候选（这些簇内所有点）
	cands := make([]int, 0)
	for _, t := range cs {
		cands = append(cands, a.lists[t.c]...)
	}

	// 3) 在候选中精确算 TopK
	type pair struct {
		id   int
		dist float32
	}
	ps := make([]pair, 0, len(cands))
	for _, id := range cands {
		ps = append(ps, pair{id: id, dist: cosineDist(q, a.data[id])})
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].dist < ps[j].dist })
	if topK > len(ps) {
		topK = len(ps)
	}
	ids := make([]int, topK)
	ds := make([]float32, topK)
	for i := 0; i < topK; i++ {
		ids[i] = ps[i].id
		ds[i] = ps[i].dist
	}
	return ids, ds
}

/************
 *   Demo   *
 ************/

// 生成一张随机 Bingo（简单：1..75 里取 25 个不重复数）
func genRandomCard(rng *rand.Rand) [25]int {
	p := rng.Perm(75)
	var c [25]int
	for i := 0; i < 25; i++ {
		c[i] = p[i] + 1
	}
	return c
}

func bruteForce(q vec, all []vec, k int) ([]int, []float32) {
	type pair struct {
		id   int
		dist float32
	}
	ps := make([]pair, 0, len(all))
	for i, v := range all {
		ps = append(ps, pair{i, cosineDist(q, v)})
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].dist < ps[j].dist })
	if k > len(ps) {
		k = len(ps)
	}
	ids := make([]int, k)
	ds := make([]float32, k)
	for i := 0; i < k; i++ {
		ids[i] = ps[i].id
		ds[i] = ps[i].dist
	}
	return ids, ds
}

func main() {
	initBases()
	rng := rand.New(rand.NewSource(randomSeed + 7))

	N := 1 << 24
	cards := make([][25]int, N)
	embs := make([]vec, N)
	for i := 0; i < N; i++ {
		cards[i] = genRandomCard(rng)
		embs[i] = cardToVec(cards[i])
	}

	// --- 构建 K-Means ANN ---
	kClusters := 64
	maxIters := 25
	buildStart := time.Now()
	index := NewKMeansANN(embs, kClusters, maxIters)
	buildElapsed := time.Since(buildStart)
	fmt.Printf("KMeans built. N=%d  k=%d  maxIters=%d  build=%v\n", N, kClusters, maxIters, buildElapsed)

	qc := cards[0]
	qc[3], qc[17] = 75, 1
	qvec := cardToVec(qc)

	topK := 5
	nprobe := 4

	t1 := time.Now()
	ids, dists := index.Search(qvec, topK, nprobe)
	annTime := time.Since(t1)

	fmt.Println("\nANN (KMeans) TopK:")
	for i := range ids {
		fmt.Printf("%2d) id=%4d  dist=%.4f  sim=%.4f\n", i+1, ids[i], dists[i], 1-dists[i])
	}
	fmt.Printf("⏱  ANN Search Time: %v (nprobe=%d)\n", annTime, nprobe)

	t2 := time.Now()
	bids, bd := bruteForce(qvec, embs, topK)
	bruteTime := time.Since(t2)

	fmt.Println("\nBruteForce TopK:")
	for i := range bids {
		fmt.Printf("%2d) id=%4d  dist=%.4f  sim=%.4f\n", i+1, bids[i], bd[i], 1-bd[i])
	}
	fmt.Printf("🐌 Brute Force Search Time: %v\n", bruteTime)

	fmt.Printf("\n⚖️  Speedup (vs brute within probed lists): %.2fx faster\n", float64(bruteTime)/float64(annTime))
}
