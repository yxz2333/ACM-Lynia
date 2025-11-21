#ifndef LYNIA_H
#define LYNIA_H

/*
MyTools 命名空间完整目录：

	other 类算法注释：
		- 状压二进制选取枚举子集
		- 数学常见数列求和公式、无穷级数、概率期望、组合数学
		- 容斥原理公式
		- 三分查找（二分式三分）
		- 二维偏序
		- 根号分治（阈值分块）
		- 单调栈
		- 区间DP
		- LIS（最长递增子序列）
		- 二进制拆分背包
		- bitset优化
		- 换根DP
		- 树上倍增
		- 期望计算
		- XOR异或操作
		- 异或哈希
		- gcd相关定理
		- exgcd与裴蜀定理
		- 质数定理：质数间隔
		- 博弈论SG函数
		- 二维前缀和
		- 三维前缀和

	Math<T> 数学工具：
		- gcd/lcm - 最大公约数/最小公倍数
		- exgcd - 扩展欧几里得
		- lowbit - 最低位1
		- fastPow - 快速幂
		- inv - 逆元
		- highPrecisionAdd - 高精度加法
		- factorsAll - 预处理所有数的因子
		- factorNumbers - 求一个数的所有因子
		- primeFactorsMAP/SET - 质因数分解
		- C - 组合数计算
		- decimalChange - 进制转换
		- division_block - 整除分块
		- prefixSumFactorNumber - 因数个数前缀和
		- fastGL - 预处理GCD表
		- PreInv - 预处理逆元表

	Combinatorics 组合数学：
		- C - 组合数计算
		- lucas - 卢卡斯定理
		- Stirling2 - 第二类斯特林数

	EulerPrime 质数处理：
		- isPrime - 质数判断
		- primeFactorsMAP/VEC - 加速质因数分解
		- factorNumbers - 加速因数分解
		- segmentSieve - 区间筛

	SegmentTree<Info> 线段树：
		- modify - 单点修改
		- query - 区间查询

	LazySegmentTree<Info, Tag> 懒标记线段树：
		- modify - 单点修改
		- modifyRange - 区间修改
		- query - 区间查询

	List<T, N> 链表：
		- erase - 删除节点
		- insert - 插入节点
		- push_back - 尾部插入
		- next_id/prev_id - 前后驱查询

	ST<T> ST表：
		- queryMin/Max - 最值查询
		- queryOR/AND - 按位或/与查询
		- queryGCD/LCM - GCD/LCM查询

	BitTree<T> 树状数组：
		- update - 单点更新
		- getsum - 前缀和查询
		- queryRange - 区间查询
		- lower_bound/upper_bound - 树状数组二分

	Manacher 回文处理：
		- 构造函数 - 计算最长回文

	DoubleHashString 双哈希：
		- get - 子串哈希值
		- lcp - 最长公共前缀
		- string_cmp - 字符串字典序比较

	SingleHashString 单哈希：
		- get - 子串哈希值
		- lcp - 最长公共前缀
		- string_cmp - 字符串字典序比较

	LCA 最近公共祖先：
		- lca - 求最近公共祖先
		- dis - 两点距离

	ModInt<T> 模数类：
		- 各种运算符重载 - 模运算

	KMP 字符串匹配：
		- kmp - 模式串匹配

	ACAutomaton AC自动机：
		- insert - 插入模式串
		- getFail - 构建失败指针
		- query - 多模式匹配

	MinCostMaxFlow<T> 费用流：
		- minCostMaxFlow - 最小费用最大流

	XorBase 线性基：
		- insert - 插入数字
		- check - 检查存在性
		- qmax/qmin - 最大最小异或值
		- query - 第k小异或值

	fast<T> 快速IO：
		- in - 快速输入
		- out/outln - 快速输出

	PersistentWeightSegmemtTree 主席树：
		- update - 版本更新
		- query_k_min/max - 第k小/大查询
		- query_bigger/smaller_num - 值域统计
		- query_last_version - 单版本区间查询

	Frac<T> 分数类：
		- 各种运算符 - 分数运算
		- to_inv - 转模逆元

	Trie01Vector/Array 01字典树：
		- insert - 插入数字
		- find_max_xor - 最大异或值
		- find_min_xor - 最小异或值
		- erase - 删除数字
		- contains - 存在性检查

	Trie 字典树：
		- insert - 插入字符串
		- startsWith - 前缀查询
		- search - 完整词查询

	Dsu 并查集：
		- find - 查找根节点
		- uoion - 合并集合

	图论算法：
		- scc_shrink - 强连通分量缩点
		- kruskal_rebuildTree - 克鲁斯卡尔重构树
		- cut_point - 割点
		- cut_edge - 割边
		- bcc_point - 点双连通分量
		- bcc_edge - 边双连通分量
		- is_bipartite_graph - 二分图判断
		- diameter_of_tree - 树的直径
		- centroid_of_tree - 树的重心
		- centroid_decomposition - 点分治

	工具函数：
		- randint - 随机数生成
*/

/*
Geo 几何计算完整目录：

	基础函数：
		- sgn(x) 浮点数符号
		- cmp(x, y) 浮点数比较
		- radians(degrees) 角度转弧度
		- point_point_dist(A, B) 两点距离
		- point_point_dist2(A, B) 两点距离平方
		- dot(A, B) 点积计算
		- cross(A, B) 叉积计算
		- vector_len(A) 向量长度
		- vector_len2(A) 向量长度平方
		- vector_vector_angle(A, B) 两向量夹角
		- vector_vector_angle_directed(A, B) 两向量夹角（带方向）
		- vector_rotate(A, rad) 向量旋转
		- vector_normal(A) 单位法向量
		- area_parallelogram 平行四边形面积
		- area_triangle 三角形面积
		- vector_vector_parallel 向量平行判断
		- vector_vector_relation 向量位置关系
		- vector_vector_angle_type 向量夹角类型

	点线关系：
		- point_line_relation 点和直线关系
		- point_segment_relation 点和线段关系
		- point_line_dis 点到直线距离
		- point_line_proj 点在直线投影
		- point_line_symmetry 点关于直线对称点
		- point_segment_dis 点到线段距离

	线线关系：
		- line_line_relation 两直线关系
		- line_line_cross_point 两直线交点
		- segment_segment_is_cross 两线段相交判断

	多边形：
		- point_polygon_relation 点和多边形关系
		- polygon_perimeter 多边形周长
		- polygon_area 多边形面积
		- polygon_center_point 多边形重心
		- convex_hull 凸包算法

	圆形：
		- point_circle_relation 点和圆关系
		- line_circle_relation 直线和圆关系
		- segment_circle_relation 线段和圆关系
		- line_cross_circle_points 直线圆交点
		- line_polygon_cross_points 直线多边形交点
		- circle_arc_area 扇形面积
		- circle_area 圆面积

	排序算法：
		- angle_polar_sort_atan2 atan2极角排序
		- angle_polar_sort_cross 叉积极角排序

	旋转卡壳：
		- farthest_point_to_point_dis 最远点对距离
		- farthest_point_to_point_dis2 最远点对距离平方

	几何类：
	Point<T> 点类：
		- polar_angle 极角计算
		- len/len2 向量长度
		- quadrant 象限判断
		- 运算符重载（+ - * / ^ == < >）

	Line<T> 直线类：
		- 多种构造函数
		- is_no_k 垂直线判断
		- k 斜率计算
		- b 截距计算

	Polygon<T> 多边形类：
		- perimeter 周长
		- area/area2 面积
		- area_directed/area2_directed 有向面积
		- winding_order 环绕顺序判断
		- polar_angle_sort_atan2/cross 极角排序

	Circle<T> 圆类：
		- area 圆面积
		- arc_area 扇形面积
*/

#include <iostream>
#include <exception>
#include <algorithm>
#include <utility>
#include <map>
#include <set>
#include <queue>
#include <string>
#include <cstring>
#include <cmath>
#include <list>
#include <stack>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include <chrono>
#include <thread> 
#include <future>
#include <cassert>
#define fa(i,op,n) for (int i = op; i <= n; i++)
#define fb(j,op,n) for (int j = op; j >= n; j--)
#define pb push_back
#define HashMap unordered_map
#define HashSet unordered_set
#define var auto
#define all(i) i.begin(), i.end()
#define all1(i) i.begin() + 1,i.end()
#define endl '\n'
#define px first
#define py second
using namespace std;
using namespace std::chrono;
using ll = long long;
using ull = unsigned long long;
using db = double;
using pii = pair<int, int>;
using pll = pair<ll, ll>;

namespace MyTools
{
	class other {
		/* [状压] 二进制选取枚举子集 
			for (int subset = mask; subset; subset = (subset - 1) & mask) {
				// 处理subset
			}

			fa(t, 1, k) {
				// 枚举两套无交集的二进制选择 总复杂度为 O(3 ^ m)
				// (这里复杂度比较难算，纯靠估算出 4^m 就完蛋了)
				fa(i, 0, tot - 1) {
					ll tmp = i ^ (tot - 1);
					for (ll j = tmp; j > 0; j = (j - 1) & tmp)
						dp[t][i ^ j] = max(dp[t][i ^ j], dp[t - 1][i] + mx[j]);
				}
			}
		*/

		/* [数学]
			- 常见数列求和公式
			- 常见无穷级数求和公式
			- 常见概率期望计算公式
			- 常见组合数学计算公式
		*/

		/* [容斥]
			|A ∪ B| = |A| + |B| - |A ∩ B|
			=> 计算至少属于集合 A 或 B 的元素数量

			|A ∪ B ∪ C| = |A| + |B| + |C| - |A ∩ B| - |A ∩ C| - |B ∩ C| + |A ∩ B ∩ C|
			=> 计算至少属于集合 A、B、C 之一的元素数量 (奇正偶负)

			|A ∩ B ∩ C| = |全集| - |反A ∪ 反B ∪ 反C|
			=> 计算满足 A、B、C 所有条件的元素数量

			|反A ∩ 反B ∩ 反C| = |全集| - |A ∪ B ∪ C|
			=> 计算不满足 A、B、C 任何条件的元素数量
		*/

		/* [三分 (二分式三分)]
			int ternary_search_max(int l, int r) {
				while (l < r) {
					int mid = (l + r) / 2;
					if (f(mid) < f(mid + 1)) {
						l = mid + 1;
					} else {
						r = mid;
					}
				}
				return l; // 最大值点
			}

			int ternary_search_min(int l, int r) {
				while (l < r) {
					int mid = (l + r) / 2;
					if (f(mid) > f(mid + 1)) {
						l = mid + 1;
					} else {
						r = mid;
					}
				}
				return l; // 最小值点
			}
		*/

		/* [偏序]
			1. 二维偏序
				给定若干个元素，每个元素有两个属性 (a, b)，需要统计满足 a_i < a_j 且 b_i < b_j（或其他不等关系）的元素对 (i, j) 的数量。

				- 常见变体也就是改改大于小于等其它不等关系
				- 常见解法为排序后树状数组求逆序对
		*/

		/* [根号分治(阈值分块)]
			根据某个阈值（通常是 √n 或类似的量级），将数据或操作分成“小块”和“大块”，分别采用不同的策略处理：
				“小块”（如 x ≤ B）：
					- 通常采用预处理、前缀和、差分、计数数组等高效方法，使得查询或更新可以在 O(1) 或 O(B) 时间内完成。
					- 也可以用暴力计算，小块用小块的算法，大块用大块的，但保证单次操作不超过 O(B ^ 2) 时间，总最高复杂度达 O(B ^ 3)。
				“大块”（如 x > B）：
					- 总情况数较少(最多 B 个)，可以采用暴力计算，单次操作可以保持 O(n) 复杂度，总最高复杂度达 O(n * B)。

			目标：使得总时间复杂度在 O(n√n) 或 O(q√n) 范围内。

			- 1e5 的题都可以想想能不能用根号分治优化，有时候会有奇效
			- 注意可能出现的爆空间问题，阈值 B 的设定也在于空间大小 (256MB => 64e6 int 数组)
		*/

		/* [单调栈]
			维护一个单调递减/递增栈，每次都把 i 放入栈，栈顶为最大/最小值
			- 为序列每个元素寻找其左/右边第一个比它大/小的元素
			- 计算序列每个元素作为最大/小值的区间范围
			- 根据以上性质，针对具有单调性的问题进行 dp 优化
		*/

		/* [区间 dp]
			关键在于 dp 状态的定义是否包括一个区间，比如 dp[i][j]：只处理 [i, j] 区间的最佳答案
			发现题目可能是 dp，且可以使用区间来还定义状态时，就可以套区间 dp 板子
			复杂度可以为 O(N ^ 2) 或 O(N ^ 3)，复杂度取决于 dp 转移是否需要中间点，也就是是否需要两区间合并

			O(N ^ 2)：
				fa(len, 2, n)                 // 区间长度
					fa(i, 1, n - len + 1) {   // 起点
						int j = i + len - 1;  // 终点
					}

			O(N ^ 3)：
				fa(len, 2, n)                 // 区间长度
					fa(i, 1, n - len + 1) {   // 起点
						int j = i + len - 1;  // 终点
						fa(k, i, j - 1) {     // 两区间合并中间点

						}
					}

			注意：一般区间长度 len 为 1 时需要自己初始化
		*/

		/* [LIS]

			// 动态规划 O(n ^ 2)
			// 最长单调增子序列
			fa(i, 1, n)
				fa(j, 1, i - 1)
				if (a[i] > a[j])
					dp[i] = max(dp[i], dp[j] + 1);
			ans = *max_element(dp + 1, dp + n + 1);

			// 二分+单调栈 O(nlogn)
			// 最长单调不增子序列
			int cnt = 0;
			st[++cnt] = a[1];
			fa(i, 2, n) {
				if (a[i] <= st[cnt])st[++cnt] = a[i];
				else {
					int id = lower_bound(st + 1, st + 1 + cnt, a[i], greater<>()) - st;
					st[id] = a[i];
				}
			}
			ans = cnt;


			计算 LIS 数量：反向算性质相反的子序列数量即可。
		*/

		/* [二进制拆分背包]
			n 种物品，每种物品有 m 个，重量为 w，价值为 v，转成 nlogn 个物品，然后 01 背包操作

			// 原物品：最多根号 N 种，根号 N 个
			//		=> 
			// 新物品 根号 N * logN 个
			var h = MT::map_to_vector(yuan); // 原物品 { 物品重量 w：物品个数 }
			var ve = vector<int>();          // 二进制拆分后新物品 { 物品新重量 nw：个数为 1 } 
			fa(i, 0, (int)h.size() - 1) {
				var[x, y] = h[i];
				for (int j = 1; j <= y; j <<= 1) { // 枚举物品个数
					y -= j;
					ve.pb(j * x);
				}
				if (y)ve.pb(y * x); // 存在余数
			}

 			// 01 背包
			var dp = vector<bool>(n + 1);
			dp[0] = 1;
			for (const var& x : ve)
				fb(i, n, x)
				dp[i] = dp[i] || dp[i - x];
		*/

		/* [bitset]
			比整数 (最多 64 位) 存的长，比 bool 数组算的快 (对整个 bitset 操作时，自带 /64 常数)
			- 布尔 DP 问题
			- 集合交并等处理问题

			成员函数：
				reset()：初始化全 0
				set()：初始化全 1
				any()：有 1 则 true
				none()：无 1 则 false
				flip()：整个 bitset 按位取反
				count()：返回 1 的个数

				_Find_first(), _Find_next()：
					for (int pos = bs._Find_first(); pos < bs.size(); pos = bs._Find_next(pos)) {
						// 遍历每个有 1 的位置
					}


			优化 bool 背包：

				优化前：O(n * m)
					dp[0] = 1;
					fa(i, 1, n)
						fb(j, W, w[i])
							dp[j] |= dp[j - w[i]];

				优化后：O(n * m / 64)
					dp[0] = 1;
					fa(i, 1, n)
						dp |= dp << w[i]; // 一次移位即可计算所有背包容量的情况
		*/

		/* [换根 dp]
			- 题目要求输出每个点为根的答案
			- 计算答案时，必须考虑到节点上方的贡献


			~  三种换根方法  ~
			
			1. 直接 down 数组公式推导，比较简单的情况。


			2. 用 up/down 数组，并用 multiset/bitset 数组 mp[u][v] 辅助。

				第一次 dfs，后根遍历(全部子节点递归完后)，计算 down 数组；
				第二次 dfs，先根遍历(全部子节点递归前 或 单个子节点递归前)，计算 up 数组。
				此方法较为通用，但复杂度较高。

				up 数组可开个二维数组 mp[N][N] (可通过哈希表加速) 求出
					=> mp[u][v]：u 的所有子树 (除了 v 子树) 的答案 (如：各子节点的集合等)
						=> 用前后缀算
						=> 全算了再减掉子树
					=> up[now] = F( mp[u][v] + up[fa] ); (F 函数代表这样之后还得进行处理，如：bitset 要 <<= 1)
					=> 当然可以再优化成一维的 mp[v]，如：multiset<int> mp[N];


			3. 只用一个 multiset 数组 dp。
				=> 要能明确 dp 里存了哪些答案，所以用 multiset。

				只用一个数组的话，要求 dp[now]: 以 now 为根节点的整个树的答案，不分上面下面。

				第一次 dfs: 直接后根遍历，按 down 数组的方式计算；
				第二次 dfs: 
					=> 在枚举到 now-to 时，now 子树撤销来自 to 子树的贡献，相当于砍掉 now-to 连边;
					=> (可选，取决于答案如何更新)撤销完所有来自 to 子树的贡献后，再求一遍 now 子树的答案(如最值什么的)；
					=> 重新更新 to 子树的答案：
						=> 撤销 to 子树之前的答案，把 now 看成 to、to 看成 now，建 to-now 连边，更新 dp；
						=> 更新完后，此时 dp 已经变成了以 to 为根的整个树的答案；
					=> (注意)拆了 now-to 更新答案？还是建完 to-now 连边更新答案？看题意决定；
					=> 子树 dfs 完后，now 子树的撤销全部还原。

		*/

		/* [树上倍增]
			- 有唯一父子关系的题都能用(一个点能到达的下一个点是唯一确定的)，如：树、基环树(n个点n条边)
			- 要考虑树上一条链的贡献的情况，且链可能是基环树上的环
			- 结合 LCA 来算两节点间的 dp，即算 merge_func( dp(i, lca(i, j)), dp(j, lca(i, j)) )
			- 更多板子详见 LCA 部分

			// 初始化倍增
			var p = Vec2<int>(n + 1, 31);  // 倍增父节点表
			var dp = Vec2<pll>(n + 1, 31); // 从节点 i 开始，向上跳跃 2 ^ j 步的路径上的信息
			fa(j, 1, 30) {
				fa(i, 1, n) {
					p[i][j] = p[p[i][j - 1]][j - 1];
					// f 函数根据题意分析
					dp[i][j] = f(dp[i][j - 1], dp[p[i][j - 1]][j - 1]);
				}
			}

			// 查询倍增
			int now; // 当前节点
			fb(i, 30, 0) {
				// 从大到小检查就行
				if (满足题目条件 and p[now][i]) {
					
					now = p[now][i]; // 跳到下一个节点
				}
			}
		*/

		/* [期望]
			- 大部分都可以推公式解决，难度在于找出最好推式子的 E(dp) 的状态，得多找多试几种 E
			- 有些是期望 dp 题，是推 E 的转移方程，通常从后往前
			- 可能用到二分或三分(凹凸函数)

			推公式例题：
				- (几何分布) 升级装备，每次升级花费 1 单位时间，成功概率 p，失败则装备等级不变，求升 1 级所需期望时间。
					- E=p⋅1+(1−p)(1+E)
					- E=1+(1−p)E
					- E=1/p

				- (加了限制的几何分布) 每次尝试做一个任务，耗时 a，成功的概率为 p，但做完后不能立刻知道成功与否，必须等 b 时间统一验证，一旦验证到成功就停止，求期望总时间。
					- E=p(a+b)+(1−p)(a+b+E)
					- E=a+b+(1−p)E
					- pE=a+b
					- E=(a+b)/p
		*/

		/* [XOR]
			1. 异或交换法
				a = a ^ b
				b = a ^ b
				a = a ^ b

			2. a ^ b = c, a ^ c = b, b ^ c = a

			3. a - b <= a ^ b <= a + b
		*/

		/* [异或哈希]
			异或哈希的精髓就是把原来的数赋个随机值，让 a XOR b 的值在值域上唯一，一般在 ull 范围内取随机数赋值。
			MT::randint<ull>(0, ULLONG_MAX)
			- 集合判等/判重问题
			- 偶数次出现问题
			- 路径/子树判重问题，节点分配哈希值
		*/

		/* [gcd]
			1. 更相减损术
				- b >= a 时：gcd(a, b) = gcd(a, b - a) = gcd(a, b % a) 
				- 一般式：gcd(a, b) = gcd(a OR b, |a - b|)
				- 推广到：{
					gcd(a, b, c, d) = gcd(a, |b - a|, |c - b|, |d - c|)
						=> gcd(a, b, c, d) = gcd(a, |b - a|, |c - a|, |d - a|)
					gcd(a, b, c, d) = gcd(|a - b|, |b - c|, |c - d|, d)
				}

			2. 辗转相除法
				gcd(a, b) = gcd(a, b % a) 

			3. gcd 题有时可以联想到分解因数，毕竟两数共有的最大因数就是 gcd
		*/

		/* [exgcd 与 裴蜀定理]
			exgcd 最基础的用法是用来求 ax + by = gcd(a, b) 的整数解
		*/

		/* [质数定理：质数间隔]
			- 小于 n 的质数个数 c(n) ~ n / ln(n)，即当 n 很大时，c(n) 约等于 n / ln(n)
			- 平均间隔 ~ ln(n)，即当 n 很大时，相邻质数的平均间隔约等于 ln(n)，间隔并不会很远，如 n = 1e18 时，ln(n) = 41
			- 最大间隔 ~ (ln(n)) ^ 2，即当 n 很大时，相邻质数的最大间隔约等于 (ln(n)) ^ 2，如果 n 较小，建议直接算
		*/

		/* [博弈论：SG函数]
			想象成 Nim 游戏，也就是取石子游戏，给你 n 堆石子堆，交给两个人交替用多种方式取石子。

			SG(x) == 0：先手必败
			SG(x) != 0：先手必胜

			单堆石子：SG(x) = mex{ SG(x 的后继状态) }
			多堆石子(每堆石子都是一个独立的子游戏)：SG_total = SG(x1) ^ SG(x2) ^ ... ^ SG(xn)

			对于经典 Nim 游戏 (多堆石子，每次可以取任意多的石子)：
				单堆石子的SG函数：对于一堆数量为 x 的石子，SG(x) = x                            <==  SG(0) = 0, SG(1) = mex{ SG(0) } = 1, SG(2) = mex{ SG(0), SG(1) } = 2...
				整个游戏的SG值：SG_total = SG(a1) ^ SG(a2) ^ ... ^ SG(an)，ai 为每堆石子数量
		*/

		/* [二维前缀和]
		 
			===== 初始化前缀和 =====

			fa(i, 1, n)
			fa(j, 1, m) {
				// 继承上一轮
				pre[i][j] = pre[i - 1][j] + pre[i][j - 1] - pre[i - 1][j - 1];

				// 加上当前贡献
				pre[i][j] += a[i][j];
			}


			// 带 0 玩的情况下，直接把有 < 0 的项删掉就行 
			fa(i, 0, n)
			fa(j, 0, m) {
				// 继承上一轮
				if (i == 0 and j == 0)continue;
				else if (i == 0)pre[i][j] = pre[i][j - 1];
				else if (j == 0)pre[i][j] = pre[i - 1][j];
				else pre[i][j] = pre[i - 1][j] + pre[i][j - 1] - pre[i - 1][j - 1];

				// 加上当前贡献
				pre[i][j] += a[i][j];
			}


			===== 二维差分 ===== 

			// 对以下矩阵每个位置 += 1
			// (x, y)             (x, y + leny - 1)
			// (x + lenx - 1, y)  (x + lenx - 1, y + leny - 1)

			cnt[x][y]++;
			cnt[x + lenx][y]--;
			cnt[x][y + leny]--;
			cnt[x + lenx][y + leny]++;

			fa(i, 1, n)fa(j, 1, m)cnt[i][j] += cnt[i - 1][j] + cnt[i][j - 1] - cnt[i - 1][j - 1];


			===== 询问二维前缀和 =====

			-- x, y, nx, ny --

			(nx, ny)  (nx, y)
			( x, ny)  ( x, y)

			pre[x][y]
			+ pre[nx - 1][ny - 1]
			- pre[nx - 1][y] - pre[x][ny - 1]
		*/

		/* [三维前缀和]

			===== 初始化前缀和 =====

			fa(i, 1, n)
			fa(j, 1, m)
			fa(k, 1, q) {
				// 继承上一轮
			    pre[i][j][k] =
					pre[i - 1][j][k] + pre[i][j - 1][k] + pre[i][j][k - 1]
					+ pre[i - 1][j - 1][k - 1] 
					- pre[i - 1][j - 1][k] - pre[i - 1][j][k - 1] - pre[i][j - 1][k - 1];

				// 加上当前贡献
				pre[i][j][k] += a[i][j][k];
			}


			// 带 0 玩的情况下，直接把有 < 0 的项删掉就行
			fa(i, 0, n)
			fa(j, 0, m)
			fa(k, 0, q) {
				// 继承上一轮
				if (i == 0 and j == 0 and k == 0)continue;
				else if (i == 0 and j == 0)pre[i][j][k] = pre[i][j][k - 1];
				else if (i == 0 and k == 0)pre[i][j][k] = pre[i][j - 1][k];
				else if (j == 0 and k == 0)pre[i][j][k] = pre[i - 1][j][k];
				else if (i == 0)pre[i][j][k] = pre[i][j - 1][k] + pre[i][j][k - 1] - pre[i][j - 1][k - 1];
				else if (j == 0)pre[i][j][k] = pre[i - 1][j][k] + pre[i][j][k - 1] - pre[i - 1][j][k - 1];
				else if (k == 0)pre[i][j][k] = pre[i - 1][j][k] + pre[i][j - 1][k] - pre[i - 1][j - 1][k];
				else {
					pre[i][j][k] = 
						pre[i - 1][j][k] + pre[i][j - 1][k] + pre[i][j][k - 1]
						+ pre[i - 1][j - 1][k - 1]
						- pre[i - 1][j - 1][k] - pre[i - 1][j][k - 1] - pre[i][j - 1][k - 1];
				}

				// 加上当前贡献
				pre[i][j][k] += a[i][j][k]; 
			}


			===== 询问三维前缀和 =====

			-- x, y, z, nx, ny, nz --

			pre[x][y][z]
			- pre[nx - 1][y][z] - pre[x][ny - 1][z] - pre[x][y][nz - 1]
			- pre[nx - 1][ny - 1][nz - 1] 
			+ pre[nx - 1][ny - 1][z] + pre[nx - 1][y][nz - 1] + pre[x][ny - 1][nz - 1]
		*/

		/* [py 随机数]
		
			1. 随机生成一棵具有n个节点的树，返回边列表

				def generate_random_tree(n):
					if n <= 1:
						return []
    
					nodes = list(range(1, n + 1))
					random.shuffle(nodes)
    
					edges = []
					for i in range(1, n):
						parent = random.randint(0, i - 1)
						u = nodes[parent]
						v = nodes[i]
						if u > v:
							u, v = v, u
						edges.append((u, v))
    
					return edges

				# 写入文件
				with open("example.txt", "w") as f:
					f.write(f"{n}\n")
					for u, v in edges:
						f.write(f"{u} {v}\n")


			2. 生成一个长度为 n 的随机排列

				def generate_permutation(n):
					perm = list(range(1, n + 1))
					random.shuffle(perm)
					return perm


			3. 生成一个长度为 n 的随机字符串
				
				import string

				def random_string(n, mode='both'):
					"""
					生成长度为 n 的随机字符串

					参数:
					n: 字符串长度
					mode: 字符类型
						- 'lower': 仅小写字母
						- 'upper': 仅大写字母
						- 'both': 大小写字母都包含
						- 'digits': 仅数字
						- 'all': 字母+数字
					"""

					if mode == 'lower':
						chars = string.ascii_lowercase
					elif mode == 'upper':
						chars = string.ascii_uppercase
					elif mode == 'both':
						chars = string.ascii_letters
					elif mode == 'digits':
						chars = string.digits
					elif mode == 'all':
						chars = string.ascii_letters + string.digits
					else:
						raise ValueError("mode 必须是 'lower', 'upper', 'both', 'digits' 或 'all'")

					return ''.join(random.choices(chars, k=n))
			

			4. 生成随机图
				
				def generate_random_graph(n, m, directed=False, connected=False):
					"""
					生成随机图

					Args:
						n: 节点数 (1到n)
						m: 边数
						directed: 是否为有向图
						connected: 是否保证连通

					Returns:
						边列表 [(u, v), ...]
					"""

					edges = set()

					# 保证连通性：先构建生成树
					if connected and n > 1:
						nodes = list(range(1, n + 1))
						random.shuffle(nodes)
						for i in range(1, n):
							u = nodes[i-1]
							v = nodes[i]
							edges.add((u, v) if u < v or directed else (min(u, v), max(u, v)))

					# 添加剩余边
					while len(edges) < m:
						u = random.randint(1, n)
						v = random.randint(1, n)
						if u == v:
							continue
						edge = (u, v) if directed else (min(u, v), max(u, v))
						edges.add(edge)

					return list(edges)[:m]

				def generate_random_weighted_graph(n, m, directed=False, connected=False, weight_range=(1, 10)):
					"""
					生成带权随机图
					"""

					edges = generate_random_graph(n, m, directed, connected)
					return [(u, v, random.randint(*weight_range)) for u, v in edges]
		*/

		/* [启发式合并]
			启发式合并是一种优化技巧，主要用于解决需要频繁合并集合的问题，可以把 O(n^2) 优化到 O(nlogn)

			- 核心思想是：在合并两个集合时，总是将较小的集合合并到较大的集合中；
						在树上的操作时，每次找重儿子就行，合并路径就相当于在重链上移动
			- 可能用到 set、map、并查集

			CF600E 部分参考代码：
				struct node {
					ll mx_sum, mx;
					map<int, ll>mp;
					vector<int>ve;
					void init(int x) {
						mp[x]++;
						ve.pb(x);
						mx = 1;
						mx_sum = x;
					}
					void add(int x) {
						if (mp[x] == mx) {
							mx++;
							mx_sum = x;
						}
						else if (mp[x] == mx - 1)mx_sum += x;
						mp[x]++;
						ve.pb(x);
					}
					int size() const {
						return ve.size();
					}
				};

				var dfs = [&](var dfs, int now, int fa)->void {
					if (g[now].size() == 1 and now != 1) {
						s[now].init(col[now]);
						ans[now] = col[now];
						return;
					}
					int mx = 0, mx_id = 0;
					for (int to : g[now]) {
						if (to == fa)continue;
						dfs(dfs, to, now);
						if (s[to].size() > mx) {
							mx = s[to].size();
							mx_id = to;
						}
					}
					s[now] = move(s[mx_id]);
					s[now].add(col[now]);
					for (int to : g[now]) {
						if (to == fa or to == mx_id)continue;
						for (int x : s[to].ve)s[now].add(x);
					}
					ans[now] = s[now].mx_sum;
				};
		*/

		/* [分层图]
			将原图复制成 k+1 层（k 是状态维度），每层表示不同的状态，层与层之间通过状态转移边连接。

			题目通常会在基础图遍历上增加额外限制条件，如：
			- 最多可以使用 k 次特殊能力（如：跳过边、改变边权等）
			- 有状态依赖的移动（如：拿到钥匙后才能通过门）

			分层的方式有很多，按编号分层、或者开多维数组；
			为了防止 TLE，可能需要动态分层、滚动数组、多次 dij 等方式实现空间优化，减少一次跑图节点数。
		*/
	
		/* [多源最短路]
			将所有起点同时加入队列初始层，一次 BFS 即可求出所有位置到最近起点的距离，O(n + m) 或 O(n)

			典型问题：
			- 多个火源同时蔓延，求每个位置被火覆盖的最早时间
			- 有多个起点，求这些起点中的两两最近距离 (同时把最短路和次短路跑完，每个起点的次短路即是该起点到最近起点的距离)

			当然边权不全是 1，那就是用多源 dij 了
		*/

		/* [曼哈顿距离]
			曼哈顿距离: dis = |x1 - x2| + |y1 - y2|
			切比雪夫距离：dis = max(|x1 - x2|, |y1 - y2|)
			曼哈顿距离的转换：
				因为 |x1 - x2| = max(x1 - x2, x2 - x1)，所以 
				dis = max({ (x1 - x2) + (y1 - y2), (x1 - x2) - (y1 - y2), ... }) 4 项
				dis = max({ (x1 + y1) + (x2 - x2), (x1 - y1) - (x2 - y2), ... }) 4 项
				因此只需要处理所有点的 (x + y) 和 (x - y) 即可
			
			原坐标系的曼哈顿距离转切比雪夫距离：(x, y) => (x + y, x - y)
			原坐标系的切比雪夫距离转曼哈顿距离：(x, y) => ((x + y) / 2, (x - y) / 2)
		*/

		/* [贪心 or tricks]
			1. [l_i, r_i] 区间题，有可能需要按 r_i 升序排
				- 以 R 升序排序，本质上就是在每一步都选择当前“最快结束”的区间
				- 如：给定若干区间 [l_i, r_i]，求出最多能选择多少个互不重叠的区间。

			2. 带(很可能是时间)顺序的问题可以考虑《倒序》能不能解决，有时候倒序能提供一个新视角
				- 如：有一系列任务，每个任务有一个截止时间 d_i 和收益 p_i。如何选择任务使得总收益最大？

			3. 贡献法，即考虑每一个元素的贡献而不是每一个整体
				- 求一个数组所有子数组的 (最大值 - 最小值) 之和。不从“子数组”的角度去枚举，而是从“每个元素”的角度去思考：对于最终答案，这个元素贡献了多少？
		*/
	};

	template <typename T>
	class Math
	{
	public:
		constexpr static T gcd(T a, T b)
		{
			return b ? gcd(b, a % b) : a;
		}
		constexpr static T lcm(T a, T b)
		{
			return a / gcd(a, b) * b;
		}
		constexpr static T exgcd(T a, T b, T& x, T& y)
		{
			if (b == 0)
			{
				x = 1;
				y = 0;
				return a;
			}
			T d = exgcd(b, a % b, x, y);
			T t = x;
			x = y;
			y = t - a / b * y;
			return d;
		}
		constexpr static T lowbit(T k)
		{
			return k & (-k);
		}
		constexpr static T fastPow(T a, T n, T mod = 1e9 + 7)
		{ // 快速幂
			T ans = 1;
			a %= mod;
			while (n)
			{
				if (n & 1)
					ans = (ans * a) % mod;
				a = (a * a) % mod;
				n >>= 1;
			}
			return ans;
		}
		constexpr static T inv(T x, T mod)
		{ // 快速幂求逆
			return fastPow(x, mod - 2, mod);
		}
		static vector<int> highPrecisionAdd(vector<int>& A, vector<int>& B)
		{ // AB倒序输入
			vector<int> C;
			int t = 0; // 进位
			for (int i = 0; i < A.size() || i < B.size(); i++)
			{
				if (i < A.size())
					t += A[i];
				if (i < B.size())
					t += B[i];
				C.push_back(t % 10);
				t /= 10;
			}
			if (t)
				C.push_back(1);
			return C; // 倒序
		}
		static vector<vector<int>> factorsAll(int n)
		{ // 计算 [1, n] 每个数的各个因子 \
			可以 O(nlogn) 计算各个数的因子，防卡 O(n根号n) 的题
			vector<vector<int>> factorsCnts(n + 1);
			for (int i = 1; i <= n; i++)
				for (int j = i; j <= n; j += i)
					factorsCnts[j].push_back(i);
			return factorsCnts;
		}
		template<const int N>
		static array<vector<int>, N> factorsAllArray() {
			static array<vector<int>, N> factorsCnts;
			for (int i = 1; i <= N - 1; i++)
				for (int j = i; j <= N - 1; j += i)
					factorsCnts[j].push_back(i);
			return factorsCnts;
		}
		static vector<int> factorNumbers(int n)
		{ // 统计n有哪些因子
			vector<int> factors;
			for (int i = 1; i <= n / i; i++)
			{
				if (n % i == 0)
				{
					factors.push_back(i);
					if (i != n / i)
						factors.push_back(n / i);
				}
			}
			return factors;
		}
		static map<T, int> primeFactorsMAP(T n)
		{ // 分解质因数
			map<T, int> mp;
			T m = n;
			for (T i = 2; i <= n / i; i++)
			{
				while (m % i == 0)
				{
					m /= i;
					mp[i]++; // 注意避免 mp 无必要的使用
				}
			}
			if (m > 1)
				mp[m]++;
			return mp;
		}
		static set<T> primeFactorsSET(T n)
		{ // 分解质因数
			set<T> se;
			T m = n;
			for (T i = 2; i <= n / i; i++)
			{
				while (m % i == 0)
				{
					m /= i;
					se.insert(i);
				}
			}
			if (m > 1)
				se.insert(m);
			return se;
		}
		static long long C(long long a, long long b, long long mod)
		{
			long ans = 1;
			for (long long i = 1; i <= b; i++)
			{
				ans = ans * (a - i + 1) % mod * fastPow(i, mod - 2, mod) % mod;
			}
			return ans;
		}
		static string decimalChange(ll x, ll num) {
			// 十进制转 num 进制

			if (num <= 1 || num >= 10) {
				throw std::invalid_argument("进制数必须在2-9之间");
			}
			if (x == 0) return "0";

			string ans;
			ll temp = x;
			// 不断除以目标进制，取余数
			while (temp > 0) {
				ll remainder = temp % num;      // 当前位的值
				ans = char(remainder + '0') + ans;  // 添加到结果前面
				temp /= num;                    // 更新商
			}
			return ans;
		}
		static vector<T> division_block(T x) {
			// 整除分块，找到 k 为何值时，x / k 会改变，O(根号x)
			// 注意本板子是下取整
			vector<T> res;
			ll r = 0;
			for (ll l = 1; l <= x; l = r + 1) {
				res.push_back(l);
				r = x / (x / l);
			}
			return res;
		}
		static ll prefixSumFactorNumber(ll x) {
			// 求 [1, x] 所有数的因数个数的总和，O(根号x)
			// sum{ factorNumber(i) } = sum{ x / i }
			// 注意：太大会爆 ll
			var tmp = division_block(x);
			tmp.pb(x + 1);
			ll sum = 0;
			fa(i, 1, tmp.size() - 1)sum += (tmp[i] - tmp[i - 1]) * (x / tmp[i - 1]);
			return sum;
		}

		template<size_t N>
		class fastGL {
			// 范围 [0, N - 1]
			// 预处理 gcd 表，然后 O(1) 求 gcd 和 lcm
		private:
			array<array<int, N>, N> G;
		public:
			fastGL() {
				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
						if (!i || !j) G[i][j] = i + j;
						else if (i < j) G[i][j] = G[j % i][i];
						else G[i][j] = G[i % j][j];
			}
			T gcd(T x, T y) {
				return G[x][y];
			}
			T lcm(T x, T y) {
				return x / G[x][y % x] * y;
			}
		};

		template<size_t N, const int mod>
		class PreInv {
			// 预处理逆元
		private:
			array<int, N> _inv;
		public:
			PreInv() {
				_inv[1] = 1;
				for (int i = 2; i <= (int)N - 1; i++)
					_inv[i] = (mod - mod / i) * 1ll * _inv[mod % i] % mod;
			}
			int inv(int n) const {
				return _inv[n];
			}
		};
	};

	class Combinatorics
	{
	private:
		ll mod;

		void init(ll N)
		{ // 组合数计算初始化
			fac[0] = 1, inv[0] = 1;
			for (ll i = 1; i <= N; i++) {
				fac[i] = fac[i - 1] * i % mod;
				inv[i] = Math<ll>::inv(fac[i], mod);
			}
		}

	public:
		vector<ll> fac, inv; // 预处理阶乘和逆

		Combinatorics(ll N, ll mod = 1e9 + 7) : mod(mod), fac(N + 1), inv(N + 1)
		{
			init(N);
		}

		ll C(ll n, ll m)
		{ // 朴素组合数计算
			if (m < 0 || n - m < 0)
				return 0;
			return fac[n] * inv[m] % mod * inv[n - m] % mod;
		}

		ll lucas(ll n, ll m)
		{ // 卢卡斯定理
			return m == 0 ? 1 % mod : lucas(n / mod, m / mod) * C(n % mod, m % mod) % mod;
		}

		ll Stirling2(ll n, ll m)
		{ // 第二类斯特灵数：将一个有 n 个元素的集合分成 m 个非空的集合，求方案数。
			if (m > n)
				return 0; // n大于m
			ll res = 0;
			for (int i = 0; i <= m; i++)
			{ // 通项公式
				ll p = Math<ll>::fastPow(i, n) * inv[i] % mod * inv[m - i] % mod;
				if ((m - i) % 2)
					res = (res + mod - p) % mod;
				else
					res = (res + p) % mod;
			}
			return res % mod;
		}
	};

	/**
	* - 质数倍数枚举优化dp     -- O(loglogN)
	* - 优化依赖 Math::factorALL 原理的 dp
	* 
	*	《2020ICPC·小米 网络选拔赛第一场 A》
	*	fa(i, 1, N) {
	*		dp[i] += mp[i];
	*		// 枚举 i 在 [1, 1e7] 的倍数 j
	*		// 质数倍数枚举法 O(loglogW)
	*		// 本质是优化 dp 转移 
	*		for (int j = 0; j < er.cnt and i * er.prime[j] <= N; j++) {
	*			int k = i * er.prime[j];
	*			dp[k] = max(dp[k], dp[i]);
	*		}
	*		ans = max(dp[i], ans);
	*	}
	*	原理类似如此：
	*		原：1 -> 4，2 -> 4
	*		现：1 -> 2，2 -> 4
	*	因为 dp[1] 在 dp[2] 中已经处理过，故此步可以省略。
	*/
	class EulerPrime
	{
	public:
		int cnt = 0;
		vector<int> prime;

		EulerPrime(int n) : vis(n + 1), MAXN(n) { init(n); }

		bool isPrime(ll n)
		{
			// 可判断 MAXN * MAXN 以内的数，O(根号 n / logn)

			if (n <= 1) return false;
			if (n <= MAXN) return !vis[n];

			for (ll p : prime) {
				if (p * p > n) break;
				if (n % p == 0) return false;
			}
			return true;
		}

		map<ll, ll> primeFactorsMAP(ll n) {
			// 欧拉筛优化分解质因数 O(logn)，可判断 MAXN * MAXN 以内的数

			map<ll, ll> mp;
			ll m = n;
			for (ll p : prime)
			{
				if (p * p > n) break;

				while (m % p == 0)
				{
					m /= p;
					mp[p]++;
				}
				if (m <= MAXN and isPrime(m) or m == 1) break;
			}
			if (m > 1)
				mp[m]++;
			return mp;
		}

		vector<pair<ll, ll>> primeFactorsVEC(ll n) {
			// 欧拉筛优化分解质因数 O(logn)，可判断 MAXN * MAXN 以内的数

			vector<pair<ll, ll>> ve;
			ll m = n;
			for (ll p : prime)
			{
				if (p * p > n) break;

				if (m % p == 0) {
					ll cnt = 0;
					while (m % p == 0) {
						m /= p;
						cnt++;
					}
					ve.push_back({ p, cnt });
				}

				if (m <= MAXN and isPrime(m) or m == 1) break;
			}
			if (m > 1)
				ve.push_back({ m, 1 });
			return ve;
		}

		vector<ll> factorNumbers(ll n) {
			// 欧拉筛优化分解因数 O(logn)，可判断 MAXN * MAXN 以内的数

			auto ps = primeFactorsVEC(n);
			auto res = vector<ll>();
			
			// 暴力枚举每个因子出现几次
			auto dfs = [&](auto dfs, int now, ll p)->void {
				if (now > (int)ps.size() - 1) {
					res.push_back(p);
					return;
				}

				ll np = p;
				dfs(dfs, now + 1, np);
				fa(i, 1, ps[now].second) {
					np *= ps[now].first;
					dfs(dfs, now + 1, np);
				}
				return;
				};
			dfs(dfs, 0, 1);

			return res;
		}

		int segmentSieve(ll l, ll r) {
			// 区间筛，算 [l, r] 里的质数个数，复杂度 O(n)

			var is_prime = vector<bool>(r - l + 1);
			int ans = 0;

			for (ll p : prime) {
				// 区间筛，枚举 p 的倍数即合数，把区间内合数全部筛去
				for (ll j = p * max(2ll, (l + p - 1) / p); j <= r; j += p) {
					is_prime[j - l] = 0;
				}
			}

			fa(i, 0, r - l)if (is_prime[i])ans++;
			return ans;
		}

	private:
		int MAXN;
		vector<bool> vis;
		void init(int n) // 欧拉筛
		{
			for (int i = 2; i <= n; i++)
			{
				if (!vis[i]) {
					prime.push_back(i);
					cnt++;
				}
				for (int j = 0; j < cnt && i * prime[j] <= n; j++)
				{
					vis[i * prime[j]] = 1; // 用最小质因数筛去
					if (i % prime[j] == 0)
						break; // 不是最小质因数
				}
			}
		}
	};

	template <class Info>
	class SegmentTree
	{ // 下标从 1 开始
#define l(p) (p << 1)
#define r(p) (p << 1 | 1)
	protected:
		int n;
		vector<Info> info;

		void init(int _n)
		{
			vector<Info> _now(_n + 1, Info());
			init(_now);
		}
		void init(vector<Info>& _init)
		{
			n = _init.size() - 1;
			info.assign(n << 2, Info()); // 为info开辟大小
			auto build = [&](auto self, int p, int l, int r)
				{
					if (l == r)
					{
						info[p] = _init[l];
						return;
					}
					int mid = l + r >> 1;
					self(self, l(p), l, mid);
					self(self, r(p), mid + 1, r);
					pushup(p);
				};
			build(build, 1, 1, n);
		}
		void pushup(int p)
		{
			info[p] = info[l(p)] + info[r(p)];
		}

	public:
		SegmentTree() : n(0) {};
		SegmentTree(int _n)
		{
			init(_n);
		}
		SegmentTree(vector<Info>& _init)
		{
			init(_init);
		}
		void modify(int p, int pl, int pr, int x, const Info& v)
		{
			if (pl == pr)
			{
				// info[p] = v;
				info[p].apply(v);
				return;
			}
			int mid = pl + pr >> 1;
			if (x <= mid)
				modify(l(p), pl, mid, x, v);
			else
				modify(r(p), mid + 1, pr, x, v);
			pushup(p);
		}
		void modify(int p, const Info& v)
		{
			modify(1, 1, n, p, v);
		}
		Info query(int L, int R, int p, int pl, int pr)
		{
			if (pl > R || pr < L)
				return Info();
			if (L <= pl && pr <= R)
				return info[p];
			int mid = pl + pr >> 1;
			return query(L, R, l(p), pl, mid) + query(L, R, r(p), mid + 1, pr);
		}
		Info query(int L, int R)
		{
			return query(L, R, 1, 1, n);
		}

		//int query1(int p, int pl, int pr, ll x) {
		//	// 返回最后一个前缀和小于 x 的位置
		//	
		//	if (info[p].sum < x)return pr + 1;
		//	if (pl == pr)return pl;

		//	int mid = pl + pr >> 1;
		//	int pos = query1(l(p), pl, mid, x);
		//	if (pos != mid + 1)return pos;
		//	else return query1(r(p), mid + 1, pr, x - info[l(p)].sum);
		//}
		//int query1(ll x) {
		//	return query1(1, 1, n, x) - 1;
		//}

		//int query2(int p, int pl, int pr, ll x) {
		//	// 返回第一个前缀和大于x的位置
		//	if (pl == pr)return pl;

		//	int mid = pl + pr >> 1;
		//	if (info[l(p)].sum > x)return Query(l(p), pl, mid, x);
		//	else return Query(r(p), mid + 1, pr, x - info[l(p)].sum);
		//}
		//int query2(ll x) {
		//	return query2(1, 1, n, x);
		//}
#undef l(p)
#undef r(p)
	};
	/* Info
		struct Info {
			Info() {};

			// 要维护的值

			void apply(const Info& v) {
				// 如何单点修改
			}
		};
		Info operator+(const Info& a, const Info& b) {
			Info c;
			return c;
		}
	*/

	template <class Info, class Tag>
	class LazySegmentTree
	{
#define l(p) (p << 1)
#define r(p) (p << 1 | 1)
	protected:
		int n;
		vector<Info> info;
		vector<Tag> tag;

		void init(int _n)
		{
			vector<Info> _now(_n + 1);
			init(_now);
		}
		void init(const vector<Info>& _init)
		{
			n = _init.size() - 1;
			info.assign(n << 2, Info()); // 为info开辟大小
			tag.assign(n << 2, Tag());     // 为tag开辟大小
			auto build = [&](auto self, int p, int l, int r)
				{
					if (l == r)
					{
						info[p] = _init[l];
						return;
					}
					int mid = l + r >> 1;
					self(self, l(p), l, mid);
					self(self, r(p), mid + 1, r);
					pushup(p);
				};
			build(build, 1, 1, n);
		}
		void pushup(int p)
		{
			info[p] = info[l(p)] + info[r(p)];
		}
		void apply(int p, const Tag& v, int len)
		{
			info[p].apply(v, len);
			tag[p].apply(v);
		}
		void pushdown(int p, int pl, int pr)
		{ // 传入pl, pr计算区间长度
			int mid = pl + pr >> 1;
			apply(l(p), tag[p], mid - pl + 1);
			apply(r(p), tag[p], pr - mid);
			tag[p] = Tag(); // 设空
		}

	public:
		LazySegmentTree() : n(0) {};
		LazySegmentTree(int _n)
		{
			init(_n);
		}
		LazySegmentTree(const vector<Info>& _init)
		{
			init(_init);
		}
		void modify(int p, int pl, int pr, int x, const Info& v)
		{ // 单点修改
			if (pl == pr)
			{
				// info[p] = v;
				info[p].apply(v);
				return;
			}
			int mid = pl + pr >> 1;
			pushdown(p, pl, pr); // 传入pl, pr计算区间长度
			if (x <= mid)
				modify(l(p), pl, mid, x, v);
			else
				modify(r(p), mid + 1, pr, x, v);
			pushup(p);
		}
		void modify(int p, const Info& v)
		{
			modify(1, 1, n, p, v);
		}
		Info query(int L, int R, int p, int pl, int pr)
		{
			if (pl > R || pr < L)
				return Info();
			if (L <= pl && pr <= R)
				return info[p];
			int mid = pl + pr >> 1;
			pushdown(p, pl, pr); // 传入pl, pr计算区间长度
			return query(L, R, l(p), pl, mid) + query(L, R, r(p), mid + 1, pr);
		}
		Info query(int L, int R)
		{
			return query(L, R, 1, 1, n);
		}
		void modifyRange(int L, int R, int p, int pl, int pr, const Tag& v)
		{ // 区间修改
			if (pl > R || pr < L)
				return;
			if (L <= pl && pr <= R)
			{
				apply(p, v, pr - pl + 1); // 传入区间长度
				return;
			}
			int mid = pl + pr >> 1;
			pushdown(p, pl, pr); // 传入pl, pr计算区间长度
			modifyRange(L, R, l(p), pl, mid, v);
			modifyRange(L, R, r(p), mid + 1, pr, v);
			pushup(p);
		}
		void modifyRange(int L, int R, const Tag& v)
		{
			return modifyRange(L, R, 1, 1, n, v);
		}

		//int query_kth(int p, int pl, int pr, ll x) {
		//	/**
		//	* 返回最大的mid满足sum[1..mid] < x
		//	*/

		//	if (info[p].sum < x)return pr + 1;
		//	if (pl == pr)return pl;
		//	pushdown(p, pl, pr);

		//	int mid = pl + pr >> 1;
		//	int pos = query_kth(l(p), pl, mid, x);
		//	if (pos != mid + 1)return pos;
		//	else return query_kth(r(p), mid + 1, pr, x - info[l(p)].sum);
		//}
		//int query_kth(ll x) {
		//	return query_kth(1, 1, n, x) - 1;
		//}

#undef l(p)
#undef r(p)
	};
	/* Info 和 Tag
		struct Tag {
			Tag() {}
			// Tag(...): is_init(0) {}

			bool is_init = 1; // 初始化时不用 tag

			// 要维护的tag

			void apply(const Tag& t) {
				// 父tag怎么传到子tag
				if (not t.is_init) {
					is_init = 0;

				}
			}
		};
		struct Info {
			Info() {};

			// 要维护的值

			void apply(const Tag& t, int len) {
				// tag怎么传到要维护的值
				if (not t.is_init) {

				}
			}

			void apply(const Info& v) {
				// 如何单点修改

			}
		};
		Info operator+(const Info& a, const Info& b) {
			Info c;
			return c;
		}
	*/

	template <class T, const int N>
	class List
	{
	public:
		array<int, N> l, r;
		int cnt = 1, tail = 0; // 时间戳、尾节点时间戳
		map<T, int> pos;       // 值对应时间戳，需保证每个数只出现一次，位置下标从 1 开始
		array<T, N> value;     // 时间戳对于值
		array<int, N> id;      // 获取列表实际每个值的下标

		List() {}

		void erase(int idx)
		{
			try
			{
				if (tail == 0)
					throw runtime_error("list size is null");
				if (idx > tail)
					throw runtime_error("remove null pos");
			}
			catch (const std::exception& e)
			{
				cerr << e.what() << endl;
			}

			l[r[idx]] = l[idx];
			r[l[idx]] = r[idx];
			tail--;
		}

		void insert(int idx, T val)
		{
			value[cnt] = val;
			pos[val] = cnt;

			// idx 位置后插入节点
			l[cnt] = idx;
			r[cnt] = r[idx];
			l[r[idx]] = cnt;
			r[idx] = cnt++;

			tail++;
		}

		void push_back(T val)
		{
			value[cnt] = val;
			pos[val] = cnt;

			// 链表尾插入节点
			l[cnt] = tail;
			r[cnt] = r[tail];
			l[r[tail]] = cnt;
			r[tail] = cnt++;

			tail++;
		}

		int next_id(int id) {
			if (r[id] == 0)return -1;
			return r[id];
		}

		int prev_id(int id) {
			if (l[id] == 0)return -1;
			return l[id];
		}

		void print_all()
		{
			int k = tail;
			for (int i = r[0]; k; i = r[i])
			{
				cout << value[i] << ' ';
				k--;
			}
			cout << endl;
		}

		void init_id() {
			// 计算列表实际每个值的下标
			int k = tail;
			int cur = 1;
			for (int i = r[0]; k; i = r[i]) {
				id[value[i]] = cur;
				k--;
				cur++;
			}
		}
	};

	template<const int N>
	struct Log2Table {
		array<int, N + 1> value;
		constexpr Log2Table() {
			value[0] = -1;
			for (int i = 1; i <= N; i++)
				value[i] = value[i / 2] + 1;
		}
	};

	template <typename T>
	class ST
	{
	public:
		enum Operator {
			_min,
			_max,
			_or,
			_and,
			_gcd,
			_lcm
		};

		ST(vector<T>& arr, Operator op) :op(op)
		{
			// arr 从 1 开始
			n = arr.size() - 1;

			if (op == _min)Min.resize(n + 1);
			if (op == _max)Max.resize(n + 1);
			if (op == _or)OR.resize(n + 1);
			if (op == _and)AND.resize(n + 1);
			if (op == _gcd)GCD.resize(n + 1);
			if (op == _lcm)LCM.resize(n + 1);

			build(arr);
		}

		T queryMin(int l, int r)
		{ // 查询最小值
			assert(op == _min);

			T s = Log2.value[r - l + 1];
			return min(Min[l][s], Min[r - (1 << s) + 1][s]);
		}

		T queryMax(int l, int r)
		{ // 查询最大值
			assert(op == _max);

			T s = Log2.value[r - l + 1];
			return max(Max[l][s], Max[r - (1 << s) + 1][s]);
		}

		T queryOR(int l, int r) {
			assert(op == _or);

			T s = Log2.value[r - l + 1];
			return OR[l][s] | OR[r - (1 << s) + 1][s];
		}

		T queryAND(int l, int r) {
			assert(op == _and);

			T s = Log2.value[r - l + 1];
			return AND[l][s] & AND[r - (1 << s) + 1][s];
		}

		T queryGCD(int l, int r) {
			assert(op == _gcd);

			T s = Log2.value[r - l + 1];
			return gcd(GCD[l][s], GCD[r - (1 << s) + 1][s]);
		}

		T queryLCM(int l, int r) {
			assert(op == _lcm);

			T s = Log2.value[r - l + 1];
			return lcm(LCM[l][s], LCM[r - (1 << s) + 1][s]);
		}

	private:
		// 21：2000000
		static Log2Table<2000000> Log2; // 使用static，防止开了多个，但是注意一定要在类外初始化
		vector<array<T, 21>>Min, Max, OR, AND, GCD, LCM;
		int n;
		Operator op;

		void build(vector<T>& arr)
		{
			/**
			* 构建ST表
			*/

			for (int i = 1; i <= n; i++)
			{
				if (op == _min)Min[i][0] = arr[i];
				if (op == _max)Max[i][0] = arr[i];
				if (op == _or)OR[i][0] = arr[i];
				if (op == _and)AND[i][0] = arr[i];
				if (op == _gcd)GCD[i][0] = arr[i];
				if (op == _lcm)LCM[i][0] = arr[i];
			}
			for (int j = 1; j <= Log2.value[n]; j++)
			{
				for (int i = 1; i + (1 << j) - 1 <= n; i++)
				{
					if (op == _min)Min[i][j] = min(Min[i][j - 1], Min[i + (1 << (j - 1))][j - 1]);
					if (op == _max)Max[i][j] = max(Max[i][j - 1], Max[i + (1 << (j - 1))][j - 1]);
					if (op == _or)OR[i][j] = OR[i][j - 1] | OR[i + (1 << (j - 1))][j - 1];
					if (op == _and)AND[i][j] = AND[i][j - 1] & AND[i + (1 << (j - 1))][j - 1];
					if (op == _gcd)GCD[i][j] = gcd(GCD[i][j - 1], GCD[i + (1 << (j - 1))][j - 1]);
					if (op == _lcm)LCM[i][j] = lcm(LCM[i][j - 1], LCM[i + (1 << (j - 1))][j - 1]);
				}
			}
		}

		T gcd(T a, T b) {
			return b == 0 ? a : gcd(b, a % b);
		}

		T lcm(T a, T b) {
			return (a / gcd(a, b)) * b;
		}
	};

	template<typename T>
	Log2Table<2000000> ST<T>::Log2;


	template <typename T>
	class BitTree
	{ // 树状数组(优化的可操作前缀和)
	public:
		BitTree(int _n) :n(_n), t(_n + 10) {}

		void update(T idx, const T& k)
		{
			for (T i = idx + 1; i <= n; i += lowbit(i))
				t[i - 1] += k; // 运算符可改
		}

		T getsum(T idx)
		{
			T res = 0;
			for (T i = idx + 1; i > 0; i -= lowbit(i))
				res += t[i - 1]; // 运算符可改
			return res;
		}

		T queryRange(int l, int r)
		{
			return getsum(r) - getsum(l - 1);
		}

		// 找到最小的 p 使得 sum[1...p] >= v
		T lower_bound(T v) {
			T sum = 0;
			int pos = -1;
			for (int i = LOGN; i >= 0; i--) {
				if (pos + (1 << i) < n && sum + t[pos + (1 << i)] < v) {
					sum += t[pos + (1 << i)];
					pos += (1 << i);
				}
			}
			return pos + 1;
		}

		// 找到最小的 p 使得 sum[1...p] > v
		T upper_bound(T v) {
			T sum = 0;
			int pos = -1;
			for (int i = LOGN; i >= 0; i--) {
				if (pos + (1 << i) < n && sum + t[pos + (1 << i)] <= v) {
					sum += t[pos + (1 << i)];
					pos += (1 << i);
				}
			}
			return pos + 1;
		}

	private:
		vector<T> t; // 下标从0开始，实际上就是前缀和数组
		int n;
		int LOGN = 23;

		T lowbit(T k) { return k & -k; }
	};

	class Manacher
	{
	public:
		vector<int> lengths; // 以每个字符为中心的最长长度回文串的半径，即 lengths[i] => [l, i, r] 回文串的长度
		int max_length = -1; // 最大回文长度

		Manacher(string str)
		{
			int n = str.size();
			init(str, n);
		};

	private:
		void init(string str, int n)
		{
			// build
			string s;
			s.assign(n * 2 + 100, ' ');
			str = ' ' + str;
			int cnt = 0;
			s[++cnt] = '~', s[++cnt] = '#';
			for (int i = 1; i <= n; i++)
				s[++cnt] = str[i], s[++cnt] = '#';
			s[++cnt] = '!';

			// solve
			int mr = 0, mid = 0;
			vector<int> p(n * 2 + 100);
			for (int i = 2; i <= cnt - 1; i++)
			{
				if (i <= mr)
					p[i] = min(p[mid * 2 - i], mr - i + 1);
				else
					p[i] = 1;
				while (s[i - p[i]] == s[i + p[i]])
					++p[i];
				if (i + p[i] > mr)
					mr = i + p[i] - 1, mid = i;

				// 更新最大回文长度
				max_length = max(max_length, p[i]);

				// 放最大回文长度
				if (s[i] != '#' && s[i] != '~' && s[i] != '!')
					lengths.push_back(p[i] - 1);
			}
			max_length--;
		}
	};

	class DoubleHashString
	{
	private:
		vector<ll> hs1, bs1, hs2, bs2;

		//const int mod1 = 2147483647;
		//const int mod2 = 1000000007;
		//int generateRandomBase(int mod) {
		//	random_device rd;
		//	mt19937 gen(rd());
		//	uniform_int_distribution<> dis(1, mod - 1);
		//	return dis(gen);
		//}
		//const int base1 = generateRandomBase(mod1);
		//const int base2 = generateRandomBase(mod2);

		const int base1 = 233;     // 第一个质数基数
		const int base2 = 131;     // 第二个质数基数
		const ll mod1 = 1e9 + 7;    // 第一个模数
		const ll mod2 = 2147483647;   // 第二个模数

	public:
		int n;

		// 初始化函数，预计算质数基数的幂次
		DoubleHashString(const string& s)
		{
			// s 下标从 0 开始，l r 查询下标从 1 开始
			n = s.size();

			hs1.resize(n + 1);
			bs1.resize(n + 1);
			hs2.resize(n + 1);
			bs2.resize(n + 1);

			hs1[0] = hs2[0] = 0;
			bs1[0] = bs2[0] = 1;

			for (int i = 1; i <= n; i++)
			{
				bs1[i] = (bs1[i - 1] * base1) % mod1;
				bs2[i] = (bs2[i - 1] * base2) % mod2;
				hs1[i] = (hs1[i - 1] * base1 + s[i - 1] - 'a' + 1) % mod1;
				hs2[i] = (hs2[i - 1] * base2 + s[i - 1] - 'a' + 1) % mod2;
			}
		}

		// 获取子串的双哈希值
		pll get(int l, int r)
		{
			ll x = ((hs1[r] - hs1[l - 1] * bs1[r - l + 1]) % mod1 + mod1) % mod1;
			ll y = ((hs2[r] - hs2[l - 1] * bs2[r - l + 1]) % mod2 + mod2) % mod2;
			return make_pair(x, y);
		}


		// 通过 哈希 + 二分 找 a b 的最长公共前缀
		// O(nlogn) 如果被卡那只能用 SA 了
		static int lcp(DoubleHashString& a, DoubleHashString& b, int l1 = 1, int r1 = -1, int l2 = 1, int r2 = -1) {
			if (r1 == -1)r1 = a.n;
			if (r2 == -1)r2 = b.n;
			int lena = r1 - l1 + 1, lenb = r2 - l2 + 1;
			int l = 1, r = min(lena, lenb);
			var check = [&](int mid)->bool {
				if (a.get(l1, l1 + mid - 1) == b.get(l2, l2 + mid - 1)) return true;
				else return false;
				};
			while (l < r) {
				int mid = l + r + 1 >> 1;
				if (check(mid))l = mid;
				else r = mid - 1;
			}
			if (not check(l))return 0;
			else return l;
		}

		// 通过 哈希 + 二分 判断两字符串 a b 的字典序大小
		// 1: a > b 
		// 0: a == b 
		// -1: a < b 
		// 注：字符串下标都从 0 开始，但是 l,r 下标都从 1 开始
		static int string_cmp(DoubleHashString& a, DoubleHashString& b, const string& s1, const string& s2, int l1, int r1, int l2, int r2) {
			// 原理是通过二分找最长公共前缀，然后比较公共前缀后的一个字符即可
			int lena = r1 - l1 + 1, lenb = r2 - l2 + 1;
			int _lcp = lcp(a, b, l1, r1, l2, r2);
			if (_lcp < min(lena, lenb)) {
				if (s1[l1 + _lcp - 1] < s2[l2 + _lcp - 1])return -1;
				else return 1;
			}
			else {
				if (lena == lenb)return 0;
				else if (lena > lenb) return 1;
				else return -1;
			}
		}

		// 注意 *字符串在后面的* 用这个函数，并传入相应长度 
		pll combineHash(pll h1, pll h2, int L2) {
			ll newH1 = (h1.first * bs1[L2] + h2.first) % mod1;
			ll newH2 = (h1.second * bs2[L2] + h2.second) % mod2;
			return make_pair(newH1, newH2);
		}
	};


	class SingleHashString
	{ // 自然溢出单哈希
	private:
		vector<ull> hs, p;

		const int base = 131;

	public:
		int n;

		SingleHashString(string s)
		{
			// s 下标从 0 开始，l r 查询下标从 1 开始
			hs.resize(s.size() + 10), p.resize(s.size() + 10);
			n = s.size();
			s = ' ' + s;
			p[0] = 1;
			for (int i = 1; i < s.size(); i++)
			{
				hs[i] = hs[i - 1] * base + s[i] - 'a' + 1;
				p[i] = p[i - 1] * base;
			}
		}

		ull get(int l, int r)
		{
			return hs[r] - hs[l - 1] * p[r - l + 1];
		}

		// 通过 哈希 + 二分 找 a b 的最长公共前缀
		// O(nlogn) 如果被卡那只能用 SA 了
		static int lcp(SingleHashString& a, SingleHashString& b, int l1 = 1, int r1 = -1, int l2 = 1, int r2 = -1) {
			if (r1 == -1)r1 = a.n;
			if (r2 == -1)r2 = b.n;
			int lena = r1 - l1 + 1, lenb = r2 - l2 + 1;
			int l = 1, r = min(lena, lenb);
			var check = [&](int mid)->bool {
				if (a.get(l1, l1 + mid - 1) == b.get(l2, l2 + mid - 1)) return true;
				else return false;
				};
			while (l < r) {
				int mid = l + r + 1 >> 1;
				if (check(mid))l = mid;
				else r = mid - 1;
			}
			if (not check(l))return 0;
			else return l;
		}

		// 通过 哈希 + 二分 判断两字符串 a b 的字典序大小
		// 1: a > b 
		// 0: a == b 
		// -1: a < b 
		// 注：字符串下标都从 0 开始，但是 l,r 下标都从 1 开始
		static int string_cmp(SingleHashString& a, SingleHashString& b, const string& s1, const string& s2, int l1, int r1, int l2, int r2) {
			// 原理是通过二分找最长公共前缀，然后比较公共前缀后的一个字符即可
			int lena = r1 - l1 + 1, lenb = r2 - l2 + 1;
			int _lcp = lcp(a, b, l1, r1, l2, r2);
			if (_lcp < min(lena, lenb)) {
				if (s1[l1 + _lcp - 1] < s2[l2 + _lcp - 1])return -1;
				else return 1;
			}
			else {
				if (lena == lenb)return 0;
				else if (lena > lenb) return 1;
				else return -1;
			}
		}

		// 注意 *字符串在后面的* 用这个函数，并传入相应长度 
		ull combineHash(ull h1, ull h2, int L2) {
			ll newH = h1 * p[L2] + h2;
			return newH;
		}
	};

	class LCA
	{
	public:
		vector<int> depth;
		vector<vector<int>> f;    // f[x][i]即x的第2^i个祖先 (31是倍增用的，最大跳跃为2^30)
		//vector<vector<ll>> dp;  // 查询两节点间信息用

		LCA(int n, vector<vector<pll>>& e)
		{
			f.assign(n + 10, vector<int>(31));
			depth.assign(n + 10, 0);
			//dp.assign(n + 10, vector<ll>(31, -1));

			init(1, 0, e);
		}

		int lca(int x, int y)
		{
			// 默认 x 节点比 y 节点深
			if (depth[x] < depth[y])swap(x, y);

			// 将 x 节点提到跟 y 节点一个深度
			for (int i = 29; i >= 0; i--)
				if (depth[x] - (1 << i) >= depth[y])
					x = f[x][i];

			if (x == y) return x; // 特判：y 就是原本 x 的祖宗

			// 说明还没找到祖宗，更新 a、b 后接着跳
			for (int i = 29; i >= 0; i--)
				if (f[x][i] != f[y][i])
					x = f[x][i], y = f[y][i];

			return f[x][0];
		}

		int dis(int x, int y) {
			// x y 两节点距离

			int f = lca(x, y);
			return depth[x] - depth[f] + depth[y] - depth[f];
		}

		tuple<ll, ll> query(int x, int y) {
			/**
			* 查询 x y 两节点路径上的聚合信息 (dp)
			* 写法跟 lca 函数一样
			* 参考代码已注释，具体修改看题意
			*/

			// 默认 x 节点比 y 节点深
			if (depth[x] < depth[y])swap(x, y);

			// 将 x 节点提到跟 y 节点一个深度
			ll res = -1;
			for (int i = 29; i >= 0; i--)
				if (depth[f[x][i]] >= depth[y]) {
					//res = max(res, dp[x][i]);
					x = f[x][i];
				}

			// 特判：y 就是原本 x 的祖宗
			if (x == y)return { res, x };

			// 说明还没找到祖宗，更新 a、b 后接着跳
			for (int i = 29; i >= 0; i--)
				if (f[x][i] != f[y][i]) {
					//res = max({ res, dp[x][i], dp[y][i] });
					x = f[x][i], y = f[y][i];
				}

			//res = max({ res, dp[x][0], dp[y][0] });

			return { res, f[x][0] };
		}

	private:
		void init(int now, int fa, vector<vector<pll>>& e)
		{
			depth[now] = depth[fa] + 1;  // 更新深度
			f[now][0] = fa;              // 第一个祖先
			for (int i = 1; (1 << i) <= depth[now]; i++) {
				f[now][i] = f[f[now][i - 1]][i - 1]; // 求 now 的各个祖先

				// 求 now->2^i祖先 的信息
				// 合并信息：now->2^(i-1)祖先 和 2^(i-1)祖先->2^i祖先
				//dp[now][i] = max(dp[now][i - 1], dp[f[now][i - 1]][i - 1]);
			}

			for (const var& [to, w] : e[now])
			{
				if (to == fa)continue;

				// 初始化从 now 到 父节点 的信息
				//dp[to][0] = w; 

				init(to, now, e);
			}
		}
	};

	template <const int T>
	class ModInt
	{
	private:
		const static int mod = T;
	public:
		long long x;

		static long long to_modint(long long x) {
			return x < 0 ? (x + mod) % mod : x % mod;
		}

		ModInt(long long x = 0) : x(to_modint(x)) {}

		operator long long() const { return x; }

		ModInt operator+(const ModInt& a)const { return to_modint(x + a.x); }
		ModInt operator-(const ModInt& a)const { return to_modint(x - a.x); }
		ModInt operator*(const ModInt& a)const { return to_modint(x * a.x); }
		ModInt operator/(const ModInt& a)const { return to_modint(x * Math<long long>::inv(a.x, mod)); }

		void operator+=(const ModInt& a) { x = to_modint(x + a); }
		void operator-=(const ModInt& a) { x = to_modint(x - a); }
		void operator*=(const ModInt& a) { x = to_modint(x * a); }
		void operator/=(const ModInt& a) { x = to_modint(x * Math<long long>::inv(a.x, mod)); }

		ModInt operator+(const long long& a)const { return to_modint(x + to_modint(a)); }
		ModInt operator-(const long long& a)const { return to_modint(x - to_modint(a)); }
		ModInt operator*(const long long& a)const { return to_modint(x * to_modint(a)); }
		ModInt operator/(const long long& a)const { return to_modint(x * to_modint(Math<long long>::inv(a, mod))); }

		void operator+=(const long long& a) { x = to_modint(x + to_modint(a)); }
		void operator-=(const long long& a) { x = to_modint(x - to_modint(a)); }
		void operator*=(const long long& a) { x = to_modint(x * to_modint(a)); }
		void operator/=(const long long& a) { x = to_modint(x * to_modint(Math<long long>::inv(a, mod))); }

		ModInt operator+(const int& a)const { return to_modint(x + to_modint(a)); }
		ModInt operator-(const int& a)const { return to_modint(x - to_modint(a)); }
		ModInt operator*(const int& a)const { return to_modint(x * to_modint(a)); }
		ModInt operator/(const int& a)const { return to_modint(x * to_modint(Math<long long>::inv(a, mod))); }

		void operator+=(const int& a) { x = to_modint(x + to_modint(a)); }
		void operator-=(const int& a) { x = to_modint(x - to_modint(a)); }
		void operator*=(const int& a) { x = to_modint(x * to_modint(a)); }
		void operator/=(const int& a) { x = to_modint(x * to_modint(Math<long long>::inv(a, mod))); }

		friend ModInt operator+(const long long& a, const ModInt& b) { return ModInt((to_modint(a) + b.x)); }
		friend ModInt operator-(const long long& a, const ModInt& b) { return ModInt(to_modint(a) - b.x); }
		friend ModInt operator*(const long long& a, const ModInt& b) { return ModInt(to_modint(a) * b.x); }
		friend ModInt operator/(const long long& a, const ModInt& b) { return ModInt(to_modint(a) * Math<long long>::inv(b.x, mod)); }

		friend ModInt operator+(const int& a, const ModInt& b) { return ModInt(to_modint(a) + b.x); }
		friend ModInt operator-(const int& a, const ModInt& b) { return ModInt(to_modint(a) - b.x); }
		friend ModInt operator*(const int& a, const ModInt& b) { return ModInt(to_modint(a) * b.x); }
		friend ModInt operator/(const int& a, const ModInt& b) { return ModInt(to_modint(a) * Math<long long>::inv(b.x, mod)); }

	};

	static int kmp(string& t, string& s)
	{ // 在文本串 t 中找到模式串 s 出现的次数

		// build
		int nn = s.size();
		vector<int> nt(nn);
		for (int i = 1; i < nn; i++)
		{
			int j = nt[i - 1];
			while (j > 0 && s[i] != s[j])
				j = nt[j - 1];
			if (s[i] == s[j])
				j += 1;
			nt[i] = j;
		}

		// kmp
		int n = t.size(), m = s.size(), j = 0;
		int last = -1e9, ans = 0;
		for (int i = 0; i < n; i++)
		{
			while (j > 0 && t[i] != s[j])
				j = nt[j - 1];
			if (t[i] == s[j])
				j += 1;
			if (j == m)
			{
				int head = i - m + 1;
				if (head >= last + m)
				{
					ans += 1;
					last = head;
				}
			}
		}
		return ans;
	}

	class KMP
	{
	private:
		vector<int> nt;
		string s;

	public:
		// 在文本串 t 中找到模式串 s 出现的次数

		// 存入模式串 s
		KMP(string& s) : s(s)
		{
			int n = s.size();
			nt.resize(n);
			for (int i = 1; i < n; i++)
			{
				int j = nt[i - 1];
				while (j > 0 && s[i] != s[j])
					j = nt[j - 1];
				if (s[i] == s[j])
					j += 1;
				nt[i] = j;
			}
		}

		// 查询文本串 t
		int kmp(string& t)
		{
			int n = t.size(), m = s.size(), j = 0;
			int last = -1e9, ans = 0;
			for (int i = 0; i < n; i++)
			{
				while (j > 0 && t[i] != s[j])
					j = nt[j - 1];
				if (t[i] == s[j])
					j += 1;
				if (j == m)
				{
					int head = i - m + 1;
					if (head >= last + m)
					{
						ans += 1;
						last = head;
					}
				}
			}
			return ans;
		}
	};

	class ACAutomaton {
	private:
		int cnt = 1;
		vector<int>in;

		struct kkk {
			int son[26] = { 0 };    // 子节点
			int fail = 0;           // 失败指针
			int flag = 0;           // 模式串起点
			int ans = 0;            // 当前节点匹配次数
			void clear() {
				memset(son, 0, sizeof(son));
				fail = flag = ans = 0;
			}
		};

		vector<kkk>trie; // Trie树

		// 拓扑排序优化
		void topu() {
			queue<int>q;
			for (int i = 1; i <= cnt; i++)
				if (!in[i])q.push(i);

			while (!q.empty()) {
				int u = q.front();
				q.pop();

				output[trie[u].flag] = trie[u].ans; // 更新输出

				int v = trie[u].fail;
				in[v]--;
				trie[v].ans += trie[u].ans;

				if (!in[v])q.push(v);
			}
		}

		int MAXN;
		int ASCII_SIZE = 26;

	public:
		vector<int>Map;                // 模式串下标 对应的 模式串起点的节点
		vector<int>output; // 模式串起点的节点 对应的 模式串在文本串中出现的个数


		// ASCII_SIZE : 26、52、128
		ACAutomaton(int MAXN, int ASCII_SIZE = 26) :MAXN(MAXN), ASCII_SIZE(ASCII_SIZE),
			in(MAXN + 10, 0), Map(MAXN + 10, 0), output(MAXN + 10, 0), trie(MAXN + 10) {}

		void clear() { // ( 这个 clear 可能有些问题 )
			for (int i = 0; i < MAXN + 5; i++) {
				Map[i] = in[i] = output[i] = 0;
				trie[i].clear();
			}
		}

		// 插入模式串
		void insert(string& s, int num) {
			int u = 1, len = s.size();
			for (int i = 0; i < len; i++) {
				int v = s[i] - ((ASCII_SIZE <= 52) ? 'a' : 0);
				if (!trie[u].son[v])trie[u].son[v] = ++cnt;
				u = trie[u].son[v];
			}
			if (!trie[u].flag)trie[u].flag = num; // 模式串起点赋值
			Map[num] = trie[u].flag;
		}

		// 构建失败指针
		void getFail() {
			queue<int>q;

			for (int i = 0; i < ASCII_SIZE; i++)trie[0].son[i] = 1;
			q.push(1);

			while (!q.empty()) {
				int u = q.front();
				q.pop();

				int Fail = trie[u].fail;

				for (int i = 0; i < ASCII_SIZE; i++) {
					int v = trie[u].son[i];
					if (!v) {
						trie[u].son[i] = trie[Fail].son[i];
						continue;
					}

					trie[v].fail = trie[Fail].son[i];
					in[trie[v].fail]++;
					q.push(v);
				}
			}
		}

		// 查询文本串
		void query(string& s) {
			int u = 1, len = s.size();
			for (int i = 0; i < len; i++)
				u = trie[u].son[s[i] - ((ASCII_SIZE <= 52) ? 'a' : 0)], trie[u].ans++;
			topu();
		}
	};

	template<typename T>
	class MinCostMaxFlow {
	public:
		// 边的结构体
		struct Edge {
			int to;   // 目标节点
			T cap;  // 边的容量
			T cost; // 边的费用
			int rev;  // 反向边在邻接表中的索引
		};

		// 构造函数，初始化节点数
		MinCostMaxFlow(int n) : n(n), graph(n + 1), dist(n + 1), prevv(n + 1), preve(n + 1), h(n + 1) {}

		// 添加边
		void addEdge(int from, int to, T cap, T cost = 0) {
			graph[from].push_back(Edge{ to, cap, cost, (int)graph[to].size() });
			graph[to].push_back(Edge{ from, 0, -cost, (int)graph[from].size() - 1 });
		}

		// 计算最小费用最大流的函数
		T minCostMaxFlow(int s, int t, T& flow) {
			T res = 0; // 最小费用
			fill(h.begin(), h.end(), 0); // 初始化势能数组
			while (true) {
				// 使用优先队列实现的Dijkstra算法
				priority_queue<pair<T, int>, vector<pair<T, int>>, greater<pair<T, int>>> pq;
				fill(dist.begin(), dist.end(), INF); // 初始化距离数组
				dist[s] = 0;
				pq.push({ 0, s });
				while (!pq.empty()) {
					auto [d, v] = pq.top();
					pq.pop();
					if (dist[v] < d) continue;
					for (int i = 0; i < graph[v].size(); ++i) {
						Edge& e = graph[v][i];
						if (e.cap > 0 && dist[e.to] > dist[v] + e.cost + h[v] - h[e.to]) {
							dist[e.to] = dist[v] + e.cost + h[v] - h[e.to];
							prevv[e.to] = v;
							preve[e.to] = i;
							pq.push({ dist[e.to], e.to });
						}
					}
				}
				if (dist[t] == INF) break; // 如果无法到达汇点，结束
				for (int i = 0; i <= n; ++i) h[i] += dist[i]; // 更新势能
				T d = INF;
				for (int v = t; v != s; v = prevv[v]) {
					d = min(d, graph[prevv[v]][preve[v]].cap); // 找到增广路径上的最小容量
				}
				flow += d; // 增加总流量
				res += d * h[t]; // 增加总费用
				for (int v = t; v != s; v = prevv[v]) {
					Edge& e = graph[prevv[v]][preve[v]];
					e.cap -= d; // 更新正向边的容量
					graph[v][e.rev].cap += d; // 更新反向边的容量
				}
			}
			return res; // 返回最小费用
		}

	private:
		int n; // 节点数
		const int INF = 1e9; // 定义无穷大
		vector<vector<Edge>> graph; // 邻接表表示的图
		vector<T> dist; // 最短路径费用
		vector<int> prevv; // 前驱节点
		vector<int> preve; // 前驱边
		vector<T> h; // 势能数组
	};

	class XorBase {
		/**
		* 线性基
		* - [常用]原序列中任意一个数都可以通过线性基里的一些数异或得到 
		* - 线性基里的任意数异或起来都不能得到 0
		* - 线性基里的数的个数唯一，在满足以上性质的前提下，存最少的数
		*/
	private:
		vector<long long> a; // 线性基基底
		const int MN = 62;
		bool flag = false;

	public:
		int rank = 0; // 秩/极大线性无关组大小，即有效向量/关键数的数量
		XorBase() :a(70, 0) {};

		void print() {
			for (int i : a)
				cout << i << ' ';
			cout << endl;
		}

		// XorBase[x]：说明第 x 位上存在代表性数 XorBase[x]
		// 这个代表数的特点就是，最高位即为 2 ^ x
		long long& operator[](int x) {
			return a[x];
		}

		// 清除基底
		void clear() {
			a.clear();
		}

		// 插入新数
		void insert(long long x) {
			// 跟 check 函数很像
			for (int i = MN; ~i; i--)
				if (x & (1ll << i))
					if (!a[i]) { a[i] = x, rank++; return; } // 线性基里没有 x，插入 x
					else x ^= a[i]; // 线性基里有 x
			flag = true;
		}

		// 插入新数
		void operator +=(long long x) {
			insert(x);
		}

		// 线性基合并
		void operator +=(XorBase& x) {
			for (int i = MN; i >= 0; i--)if (x[i])*this += x[i];
		}
		friend XorBase operator +(XorBase& x, XorBase& y) {
			XorBase z = x;
			for (int i = 62; i >= 0; i--)if (y[i])z += y[i];
			return z;
		}

		// 查是否存在线性基内
		bool check(long long x) {
			for (int i = MN; ~i; i--)
				if (x & (1ll << i))
					if (!a[i])return false;
					else x ^= a[i];
			return true;
		}

		// 查最大
		long long qmax(long long res = 0) {
			for (int i = MN; ~i; i--)
				res = max(res, res ^ a[i]);
			return res;
		}

		// 查最小
		long long qmin() {
			if (flag)return 0;
			for (int i = 0; i <= MN; i++)
				if (a[i])return a[i];
		}

		// 查询第 k 小
		long long query(long long k) {
			long long tmp[70] = { 0 };
			long long res = 0; int cnt = 0;
			k -= flag; if (!k)return 0;
			for (int i = 0; i <= MN; i++) {
				for (int j = i - 1; ~j; j--)
					if (a[i] & (1ll << j))a[i] ^= a[j];
				if (a[i])tmp[cnt++] = a[i];
			}
			if (k >= (1ll << cnt))return -1;
			for (int i = 0; i < cnt; i++)
				if (k & (1ll << i))res ^= tmp[i];
			return res;
		}
	};

	template<typename T>
	class fast {
	public:
		// 使用 __int128 直接传入 T = __int128 就行
		inline static T in()
		{
			T x = 0, f = 1;
			char ch = getchar();
			while (ch < '0' || ch>'9')
			{
				if (ch == '-')
					f = -1;
				ch = getchar();
			}
			while (ch >= '0' && ch <= '9')
				x = x * 10 + ch - '0', ch = getchar();
			return x * f;
		}
		static void out(T x)
		{
			if (x < 0)
				putchar('-'), x = -x;
			if (x > 9)
				out(x / 10);
			putchar(x % 10 + '0');
			return;
		}
		static void outln(T x)
		{
			out(x);
			putchar('\n');
		}
	};

	class PersistentWeightSegmemtTree {
		/**
		* 可持久化权值线段树(主席树)
		* 下标从 1 开始
		*
		* 注意：
		* - 本质是权值线段树，所有的思考应该建立在权值线段树上。
		* - 运用了前缀和差分原理，故只适用于静态区间，动态区间需要树状数组套主席树
		* - 若值域过大，可不开离散化，靠动态开点 (update函数)
		*/

		/* 求第 k 小模板
			int n, m; cin >> n >> m;
			var a = vector<int>(n + 1);
			var b = vector<int>();

			fa(i, 1, n)cin >> a[i], b.pb(a[i]);

			sort(all(b));
			b.erase(unique(all(b)), b.end());

			int len = b.size();
			MT::PersistentWeightSegmemtTree sg(len, n);

			var find = [&](int value)->int {
				return upper_bound(all(b), value) - b.begin();
				};

			// 初始化主席树，记得赋值
			fa(i, 1, n) sg.root[i] = sg.update(find(a[i]), 1, sg.root[i - 1]);

			while (m--) {
				int l, r, k; cin >> l >> r >> k;
				// 求区间相减，传入 l - 1
				cout << b[sg.query_k_min(l - 1, r, k) - 1] << endl;
			}
		*/

		/* 求区间多少不同元素模板
			int n; cin >> n;
			MT::PersistentWeightSegmemtTree sg(n, n);
			var mp = HashMap<int, int>();
			fa(i, 1, n) {
				int x; cin >> x;
				if (mp.count(x)) {
					// 如果之前出现过，且出现在 mp[x] 位置，
					// 就删掉之前出现的，转而把这个值放在现在的 i 上。

					// 在这个过程中，把 -1 修改的版本转移到当前版本 root[i] 上，
					// 从而表示到了 root[i] 版本才能让 mp[x] 位置的 sum -= 1

					int t = sg.update(mp[x], -1, sg.root[i - 1]);
					sg.root[i] = sg.update(i, 1, t);
				}
				else sg.root[i] = sg.update(i, 1, sg.root[i - 1]);
				mp[x] = i;
			}
			int m; cin >> m;
			while (m--) {
				int l, r; cin >> l >> r;
				cout << sg.query_last_version(l, r) << endl;
			}
		*/

	private:
		struct tree { int l = 0, r = 0; ll sum = 0; }; // 树上的节点
		int tot = 0;    // 记录节点个数，并作为版本编号
		vector<tree>t;  // 存树各个的节点 
		int len;        // 离散化后的数的个数\
												也是值域大小 (可不开离散化，纯靠动态开点即可)

		int build(int l, int r) { // 初始化建树
			int node = ++tot;
			if (l == r)return node;
			int mid = l + r >> 1;
			t[node].l = build(l, mid);
			t[node].r = build(mid + 1, r);
			return node;
		}
	public:
		vector<int>root; // 存各个版本线段树的根节点编号

		PersistentWeightSegmemtTree(int len, int n, bool f = 1) :len(len), t((n << 5) + 10), root(n + 10) {
			/**
			* m：离散化后长度(值域大小)，不用 +1 (也可不开离散化，m 本身不会开空间)
			* n：原始数组长度，不用 +1
			* f：是否开离散化，不开就不用设叶子节点，纯靠动态开点
			*/
			if (f)root[0] = build(1, len);
		}

		int update(int l, int r, int pos, int value, int pre) {
			int node = ++tot;
			t[node] = t[pre];
			t[node].sum += value;
			if (l == r)return node;
			int mid = l + r >> 1;
			if (pos <= mid)  // 新的左子节点会继承前一个版本的左子节点进行更新
				t[node].l = update(l, mid, pos, value, t[pre].l);
			else             // 同理
				t[node].r = update(mid + 1, r, pos, value, t[pre].r);
			return node;
		}

		int update(int pos, int value, int pre) {
			/**
			* 更新权值线段树 (原有的值上 += value)，加入新的叶子节点
			* 需传入 pre，即上一个版本的线段树的根节点
			*/
			return update(1, len, pos, value, pre);
		}

		int change(int l, int r, int pos, int value, int pre) {
			int node = ++tot;
			t[node] = t[pre];
			t[node].sum = value;
			if (l == r)return node;
			int mid = l + r >> 1;
			if (pos <= mid)
				t[node].l = change(l, mid, pos, value, t[pre].l);
			else
				t[node].r = change(mid + 1, r, pos, value, t[pre].r);
			return node;
		}

		int change(int pos, int value, int pre) {
			/**
			* 更新权值线段树 (原有的值改为 value)，加入新的叶子节点
			* 需传入 pre，即上一个版本的线段树的根节点
			*/
			return change(1, len, pos, value, pre);
		}

		int query_k_min(int u, int v, int l, int r, int k) {
			// 线段树版本区间 [u, v] 代替目标区间 [L, R]

			int mid = l + r >> 1;

			// 通过区间减法得左儿子中存的数值个数
			// lnum 为 原数组区间 [l, mid] 中数的个数 (即离散化数组中，值域是 [l, mid] 的数的个数)
			int lnum = t[t[v].l].sum - t[t[u].l].sum;

			if (l == r)return l;
			if (k <= lnum) // 第 k 小在左子
				return query_k_min(t[u].l, t[v].l, l, mid, k); // 线段树版本都改为 l，往左区间 [l, mid] 找
			else           // 第 k 小在右子，并注意相减
				return query_k_min(t[u].r, t[v].r, mid + 1, r, k - lnum); // 线段树版本都改为 r，往右区间 [mid + 1, r] 找
		}

		int query_k_min(int L, int R, int k) {
			/**
			* 求区间第 k 小
			* 原理是前缀和，要传入 L - 1
			* 本质就是用 R 版本线段树 减掉 L - 1 版本线段树，得到 [L, R] 区间，然后再在这个区间里找第 k 小
			*/
			return query_k_min(root[L], root[R], 1, len, k);
		}

		int query_k_max(int u, int v, int l, int r, int k) {
			// 线段树版本区间 [u, v] 代替目标区间 [L, R]

			int mid = l + r >> 1;

			// 通过区间减法得右儿子中存的数值个数
			// rnum 为 原数组区间 [mid + 1, r] 中数的个数 (即离散化数组中，值域是 [mid + 1, r] 的数的个数)
			int rnum = t[t[v].r].sum - t[t[u].r].sum;

			if (l == r)return l;
			if (k <= rnum) // 第 k 大在右子
				return query_k_max(t[u].r, t[v].r, mid + 1, r, k); // 线段树版本都改为 r，往右区间 [mid + 1, r] 找
			else           // 第 k 大在左子，并注意相减
				return query_k_max(t[u].l, t[v].l, l, mid, k - rnum); // 线段树版本都改为 l，往左区间 [l, mid] 找
		}

		int query_k_max(int L, int R, int k) {
			/**
			* 求区间第 k 大
			* 原理是前缀和，要传入 L - 1
			* 本质就是用 R 版本线段树 减掉 L - 1 版本线段树，得到 [L, R] 区间，然后再在这个区间里找第 k 大
			*/
			return query_k_max(root[L], root[R], 1, len, k);
		}

		ll query_bigger_num(int L, int R, ll k) {
			/**
			* 双版本差分查询区间值域大于等于 k 的数的总数 sum
			* L, R 为线段树版本，要传入 L - 1
			* 离散化：k 是要找的特定数在离散化数组中的下标，即 k = find(x)
			* 非离散化：k 就是要查的特定数，即 k = x
			* (考虑到值域限制，使用本函数很可能不能使用离散化，例题：2020icpc昆明)
			*/

			var query = [&](var query, int u, int v, int l, int r, ll k)->ll {
				// (原理可能有点难以理解，建议画两个版本的线段树然后看图思考)

				if (l == r) return t[v].sum - t[u].sum;
				int mid = l + r >> 1;
				if (k > mid)return query(query, t[u].r, t[v].r, mid + 1, r, k);
				else return t[t[v].r].sum - t[t[u].r].sum + query(query, t[u].l, t[v].l, l, mid, k);
				};
			return query(query, root[L], root[R], 1, len, k);
		}

		ll query_smaller_num(int L, int R, ll k) {
			/**
			* 双版本差分查询区间值域小于等于 k 的数的总数 sum
			* L, R 为线段树版本，要传入 L - 1
			* 离散化：k 是要找的特定数在离散化数组中的下标，即 k = find(x)
			* 非离散化：k 就是要查的特定数，即 k = x
			* (考虑到值域限制，使用本函数很可能不能使用离散化，例题：2020icpc昆明)
			*/

			var query = [&](var query, int u, int v, int l, int r, ll k)->ll {
				// (原理可能有点难以理解，建议画两个版本的线段树然后看图思考)

				if (l == r) return t[v].sum - t[u].sum;
				int mid = l + r >> 1;
				if (k <= mid)return query(query, t[u].l, t[v].l, l, mid, k);
				else return t[t[v].l].sum - t[t[u].l].sum + query(query, t[u].r, t[v].r, mid + 1, r, k);
				};
			return query(query, root[L], root[R], 1, len, k);
		}

		ll query_last_version(int L, int R, int v = -1) {
			/**
			* 单版本区间查询，实际上会也就是带上 v 前面所有的版本进行查询
			* v 为线段树版本，版本为 root[1, R]
			*/
			if (v == -1)v = root[R]; // 默认区间最后一个版本

			var query = [&](var query, int l, int r, int L, int R, int pre)->ll {
				// 只涉及一个版本，故目标区间不是线段树版本，而是 [L, R]

				if (L <= l and r <= R) return t[pre].sum;
				int mid = l + r >> 1;
				int res = 0;
				if (L <= mid)res += query(query, l, mid, L, R, t[pre].l);
				if (R > mid)res += query(query, mid + 1, r, L, R, t[pre].r);
				return res;
				};
			return query(query, 1, len, L, R, v);
		}

		//// 主席树区间查询实例参考
		//vector<pair<int, int>> query_frequent_numbers(int L, int R, int B) {
		//	/**
		//	* 查询区间 [L, R] 中出现次数超过 B 的数及其出现次数
		//	* L, R 为线段树版本，要传入 L - 1
		//	* 返回 vector<pair<值域下标, 出现次数>>
		//	*/
		//	vector<pair<int, int>> res;

		//	auto dfs = [&](auto dfs, int u, int v, int l, int r) {
		//		ll sum = t[v].sum - t[u].sum;

		//		if (sum <= B) return;

		//		if (l == r) {
		//			// 到达叶子节点，如果出现次数 > B，加入结果
		//			if (sum > B)res.emplace_back(l, sum);
		//			return;
		//		}

		//		int mid = l + r >> 1;
		//		dfs(dfs, t[u].l, t[v].l, l, mid);
		//		dfs(dfs, t[u].r, t[v].r, mid + 1, r);
		//		};

		//	dfs(dfs, root[L], root[R], 1, len);
		//	return res;
		//}
	};

	template<typename T = long long>
	class Frac
	{
	private:
		T abs(const T& x)const { return x < 0 ? -x : x; }
		T gcd(const T& x, const T& y)const { return y ? gcd(y, x % y) : x; }
		Frac reduce()
		{
			bool flag = 0;
			if (a < 0 && b < 0) a = -a, b = -b;
			if (a < 0) a = -a, flag = 1;
			if (b < 0) b = -b, flag = 1;
			T ggcd = gcd(a, b);
			a /= ggcd;
			b /= ggcd;
			if (flag) a = -a;
			return *this;
		}
		void swap() { std::swap(a, b); }
		Frac _swap(const Frac& t)const { return Frac(t.b, t.a); }
		T FastPow(T x, T p, T mod)const
		{
			T ans = 1, bas = x;
			for (; p; bas = bas * bas % mod, p >>= 1)
				if (p & 1) ans = ans * bas % mod;
			return ans;
		}
	public:
		T a, b;
		Frac(T A = 0, T B = 1) { a = A, b = B; }
		T to_inv(const T& mod = 1e9 + 7)const { return a * FastPow(b, mod - 2, mod) % mod; }
		Frac abs()const { return Frac(abs(a), abs(b)); }
		friend ostream& operator<<(ostream& out, const Frac& a) { out << a.a << ' ' << a.b; return out; }
		Frac operator =(const Frac& t) { return a = t.a, b = t.b, t; }
		bool operator ==(const Frac& t)const { Frac A(*this), B(t); return (A.reduce().a == B.reduce().a) && (A.b == B.b); }
		bool operator !=(const Frac& t)const { Frac A(*this), B(t); return (A.a != B.a) || (A.b != B.b); }
		bool operator >(const Frac& t)const { Frac A(*this), B(t); T ggcd = gcd(A.reduce().b, B.reduce().b); return B.b / ggcd * A.a > A.b / ggcd * B.a; }
		bool operator <(const Frac& t)const { Frac A(*this), B(t); T ggcd = gcd(A.reduce().b, B.reduce().b); return B.b / ggcd * A.a < A.b / ggcd * B.a; }
		bool operator >=(const Frac& t)const { Frac A(*this), B(t); T ggcd = gcd(A.reduce().b, B.reduce().b); return B.b / ggcd * A.a >= A.b / ggcd * B.a; }
		bool operator <=(const Frac& t)const { Frac A(*this), B(t); T ggcd = gcd(A.reduce().b, B.reduce().b); return B.b / ggcd * A.a <= A.b / ggcd * B.a; }
		Frac operator +(const Frac& t)const { T ggcd = gcd(b, t.b); return Frac(b / ggcd * t.a + t.b / ggcd * a, b / ggcd * t.b).reduce(); }
		Frac operator +=(const Frac& t) { return *this = *this + t; }
		Frac operator *(const Frac& t)const { return Frac(a * t.a, b * t.b).reduce(); }
		Frac operator *=(const Frac& t) { return *this = *this * t; }
		Frac operator -(const Frac& t)const { return (*this + Frac(-t.a, t.b)).reduce(); }
		Frac operator -=(const Frac& t) { return *this = *this - t; }
		Frac operator /(const Frac& t)const { return (t._swap(t) * (*this)).reduce(); }
		Frac operator /=(const Frac& t) { return *this = *this / t; }
		Frac operator -()const { return Frac(-a, b); }
	};


	namespace Geo {
		template<typename T>
		class Point;

		template<typename T>
		class Line;

		template<typename T>
		class Polygon;

		template<typename T>
		class Circle;

		template<typename T>
		using Vector = Point<T>;

		template<typename T>
		using Segment = Line<T>;

		template<typename T>
		using PointSet = Polygon<T>;


		const double eps = 1e-9;
		const double PI = acos(-1.0);

		template<typename T>
		int sgn(T x) {
			/**
			* 浮点数 x 的符号
			*/
			if (abs(x) < eps) return 0;
			return x > 0 ? 1 : -1;
		}

		template<typename T>
		int cmp(T x, T y) {
			/**
			* 比较两个浮点数
			* 返回值：
			*	0：x == y
			*	1: x > y
			*	-1：x < y
			*/
			if (abs(x - y) < eps)return 0;
			else return x < y ? -1 : 1;
		}

		double radians(double degrees) {
			/**
			* 角度转弧度
			*/
			return degrees * PI / 180.0;
		}

		template<typename T>
		double point_point_dist(const Point<T>& A, const Point<T>& B) {
			/**
			* 两点距离
			*/
			return sqrt((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y));
		}

		template<typename T>
		long long point_point_dist2(const Point<T>& A, const Point<T>& B) {
			/**
			* 两点距离的平方
			*/
			return (A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y);
		}

		template<typename T>
		T dot(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 计算点积  a · b = |a| |b| cos
			* - 判断两向量夹角是否 > 90
			*/
			return A.x * B.x + A.y * B.y;
		}

		template<typename T>
		T cross(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 计算叉积  a · b = |a| |b| sin
			* 逆时针输入 A B
			* - 判断两向量的相对方向
			* - 算两向量形成的平行四边形的有向面积
			* - 判断两向量夹角是否 > 180
			*/
			return A.x * B.y - A.y * B.x;
		}

		// 为了过编译，注释此处
		//Point<db> angle_to_point(const db& ang) {
		//	/**
		//	* 极角变单位坐标
		//	*/
		//	return { cos(ang), sin(ang) };
		//}

		template<typename T>
		double vector_len(const Vector<T>& A) {
			/**
			* 向量长度
			*/
			return sqrt(dot(A, A));
		}

		template<typename T>
		T vector_len2(const Vector<T>& A) {
			/**
			* 向量长度的平方
			*/
			return dot(A, A);
		}

		template<typename T>
		double vector_vector_angle(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 两向量夹角 (弧度制)
			*/
			double tt = (double)dot(A, B) / vector_len(A) / vector_len(B);
			if (tt < -1.0)
				tt = -1.0;
			if (tt > 1.0)
				tt = 1.0;
			return acos(tt);
		}


		template<typename T>
		double vector_vector_angle_directed(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 两向量夹角 (弧度制) 带方向
			* A -> B 逆时针旋转角度
			* 可以算多边形内角
			*/
			long long aa = cross(A, B);
			double ang2 = vector_vector_angle(A, B);
			int choice = sgn(aa);
			if (choice >= 0)return ang2;
			else return 2 * PI - ang2;
		}

		template<typename T>
		Vector<T> vector_rotate(const Vector<T>& A, double rad) {
			/**
			* 向量旋转 (弧度制)
			* 特殊情况是旋转90度：
			* 逆时针旋转90度：Rotate(A, pi/2)，返回Vector(-A.y, A.x)；
			* 顺时针旋转90度：Rotate(A, -pi/2)，返回Vector(A.y, - A.x)。
			*/
			return Vector<T>(A.x * cos(rad) - A.y * sin(rad), A.x * sin(rad) + A.y * cos(rad));
		}

		template<typename T>
		Vector<T> vector_normal(const Vector<T>& A) {
			/**
			* 单位法向量
			* 有时需要求单位法向量，即逆时针转90度，然后取单位值。
			*/
			return Vector<T>(-A.y / vector_len(A), A.x / vector_len(A));
		}

		template<typename T>
		T area_parallelogram(const Point<T>& A, const Point<T>& B, const Point<T>& C) {
			/**
			* 计算两向量构成的平行四边形有向面积
			* 三个点A、B、C，以 A 为公共点，得到 2 个向量 AB 和 AC，它们构成的平行四边形
			* 逆时针输入 B C
			*/
			return cross(B - A, C - A);
		}

		template<typename T>
		T area_parallelogram(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 计算两向量构成的平行四边形有向面积
			* 两个有公共点的向量 A B 构成的平行四边形
			* 逆时针输入 A B
			*/
			return cross(A, B);
		}

		template<typename T>
		double area_triangle(const Point<T>& A, const Point<T>& B, const Point<T>& C) {
			/**
			* 计算两向量构成的三角形有向面积
			* 三个点A、B、C，以 A 为公共点，得到 2 个向量 AB 和 AC，它们构成的三角形
			* 逆时针输入 B C
			*/
			return cross(B - A, C - A) / 2.0;
		}

		template<typename T>
		double area_triangle(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 计算两向量构成的三角形有向面积
			* 两个有公共点的向量 A B 构成的三角形
			* 逆时针输入 A B
			*/
			return cross(A, B) / 2.0;
		}

		template<typename T>
		bool vector_vector_parallel(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 两个向量是否平行或重合
			*/
			return sgn(cross(A, B)) == 0;
		}

		template<typename T>
		int vector_vector_relation(const Vector<T>& v1, const Vector<T>& v2) {
			/**
			* 两向量的位置关系
			* 返回值：
			*	0：v1 与 v2 共线
			*	1：v2 在 v1 的逆时针方向
			*	2：v2 在 v1 的顺时针方向
			*/

			int sign = sgn(cross(v1, v2));
			if (sign == 0)return 0;
			if (sign > 0)return 1;
			if (sign < 0)return 2;
		}

		template<typename T>
		int vector_vector_angle_type(const Vector<T>& v1, const Vector<T>& v2) {
			/**
			* 两向量夹角类型
			* 返回值：
			*	0：夹角度为 0
			*	1：夹角为锐角
			*	2：夹角为钝角
			*	3：夹角为平角，即方向相反
			*/

			var _dot = dot(v1, v2);
			if (vector_vector_relation(v1, v2) == 0) {
				// 两向量共线
				if (sgn(_dot) > 0)return 0;
				else return 3;
			}
			else {
				if (sgn(_dot) > 0)return 1;
				else return 2;
			}
		}

		template<typename T>
		int point_line_relation(const Point<T>& p, const Line<T>& v) {
			/**
			* 点和直线的位置关系
			* 返回值：
			*	1 ：p 在 v 的左边
			*	2 ：p 在 v 的右边
			*	0 ：p 在 v 上
			*/

			int c = sgn(cross(p - v.p1, v.p2 - v.p1));
			if (c < 0)return 1;
			if (c > 0)return 2;
			return 0;
		}

		template<typename T>
		bool point_segment_relation(const Point<T>& p, const Line<T>& v) {
			/**
			* 点和线段的位置关系
			* 返回值：
			*	0：p 点不在线段 v 上
			*   1：p 点在线段 v 上
			*/

			// 前者为 True 说明 p 和 线段 v 的一个端点连边，和 v 本身的夹角为 0，即 p 在 直线 v 上
			// 后者为 True 说明 p 和两端点形成平角，也就是说 p 在两端点之间
			return sgn(cross(p - v.p1, v.p2 - v.p1)) == 0 && sgn(dot(p - v.p1, p - v.p2)) <= 0;
		}

		template<typename T>
		double point_line_dis(const Point<T>& p, const Line<T>& v) {
			/**
			* 点到直线的距离
			*
			* 实际上是算了 p 和 v 的一个端点连边，然后和 v 形成的平行四边形的面积，除底得到
			*/

			return fabs(cross(p - v.p1, v.p2 - v.p1)) / point_point_dist(v.p1, v.p2);
		}

		template<typename T>
		Point<T> point_line_proj(const Point<T>& p, const Line<T>& v) {
			/**
			* 点在直线上的投影点
			*/

			double k = dot(v.p2 - v.p1, p - v.p1) / vector_len2(v.p2 - v.p1);
			return v.p1 + (v.p2 - v.p1) * k;
		}

		template<typename T>
		Point<T> point_line_symmetry(const Point<T>& p, const Line<T>& v) {
			/**
			* 点关于直线的对称点
			*/
			Point<T> q = point_line_proj(p, v);
			return Point<T>(2 * q.x - p.x, 2 * q.y - p.y);
		}

		template<typename T>
		double point_segment_dis(const Point<T>& p, const Segment<T>& v) {
			/**
			* 点到线段的距离
			*
			* 先检查点 p 到线段 v 的投影是否在线段 v 上，即看 p 和 v 的两端点的连边 pa、pb 与 v 夹角是否 > 90
			* 如果在，就直接返回点 p 到直线 v 距离
			* 如果不在，就返回线段 v 两端点到 p 点的最小距离
			*/

			if (sgn(dot(p - v.p1, v.p2 - v.p1)) < 0 || sgn(dot(p - v.p2, v.p1 - v.p2)) < 0)
				return min(point_point_dist(p, v.p1), point_point_dist(p, v.p2));
			return point_line_dis(p, v);
		}

		template<typename T>
		int line_line_relation(const Line<T>& v1, const Line<T>& v2) {
			/**
			* 两条直线的位置关系
			* 返回值：
			*	0: 平行
			*	1: 重合
			*	2: 相交
			*/

			if (sgn(cross(v1.p2 - v1.p1, v2.p2 - v2.p1)) == 0) {
				if (point_line_relation(v1.p1, v2) == 0) return 1;
				else return 0;
			}
			return 2;
		}

		template<typename T>
		Point<T> line_line_cross_point(const Point<T>& a, const Point<T>& b, const Point<T>& c, const Point<T>& d) {
			/**
			* 两条直线的交点
			* 输入 4 个点，组成两条直线 Line1 : ab, Line2 : cd
			*/

			double s1 = cross(b - a, c - a);
			double s2 = cross(b - a, d - a);
			return Point<T>(c.x * s2 - d.x * s1, c.y * s2 - d.y * s1) / (s2 - s1);
		}

		template<typename T>
		Point<T> line_line_cross_point(const Line<T>& x, const Line<T>& y) {
			/**
			* 两条直线的交点
			* 输入 2 条直线，Line1 : ab, Line2 : cd
			*/

			auto a = x.p1;
			auto b = x.p2;
			auto c = y.p1;
			auto d = y.p2;
			double s1 = cross(b - a, c - a);
			double s2 = cross(b - a, d - a);
			return Point<T>(c.x * s2 - d.x * s1, c.y * s2 - d.y * s1) / (s2 - s1);
		}

		template<typename T>
		bool segment_segment_is_cross(const Point<T>& a, const Point<T>& b, const Point<T>& c, const Point<T>& d) {
			/**
			* 两个线段是否相交
			* 输入两个线段的总计 4 个端点，Line1 : ab, Line2 : cd
			* 返回值：
			*	1：相交
			*	0：不相交
			*/

			double c1 = cross(b - a, c - a), c2 = cross(b - a, d - a);
			double d1 = cross(d - c, a - c), d2 = cross(d - c, b - c);
			return sgn(c1) * sgn(c2) < 0 && sgn(d1) * sgn(d2) < 0;  // 1: 相交；0: 不相交
		}

		template<typename T>
		bool segment_segment_is_cross(const Line<T>& x, const Line<T>& y) {
			/**
			* 两个线段是否相交
			* 输入两个线段的总计 4 个端点，Line1 : ab, Line2 : cd
			* 返回值：
			*	1：相交
			*	0：不相交
			*/

			auto a = x.p1;
			auto b = x.p2;
			auto c = y.p1;
			auto d = y.p2;
			double c1 = cross(b - a, c - a), c2 = cross(b - a, d - a);
			double d1 = cross(d - c, a - c), d2 = cross(d - c, b - c);
			return sgn(c1) * sgn(c2) < 0 && sgn(d1) * sgn(d2) < 0;  // 1: 相交；0: 不相交
		}

		template<typename T>
		int point_polygon_relation(const Point<T>& pt, const Polygon<T>& p) {
			/**
			* 点和多边形的关系
			* 输入点 pt、多边形 p
			* 返回值：
			*	0：点在多边形外部
			*	1：点在多边形内部
			*	2：点在多边形的边上
			*	3：点在多边形的顶点上
			*/

			int n = p.size();
			for (int i = 0; i < n; i++)   // 枚举点
				if (p[i] == pt) return 3;
			for (int i = 0; i < n; i++) { // 枚举边
				Line<T> v = Line<T>(p[i], p[(i + 1) % n]);
				if (point_segment_relation(pt, v)) return 2;
			}

			// 通过射线法计算点是否在多边形内部 (具体原理可以看书)
			int num = 0;
			for (int i = 0; i < n; i++) {
				int j = (i + 1) % n;
				int c = sgn(cross(pt - p[j], p[i] - p[j]));
				int u = sgn(p[i].y - pt.y);
				int v = sgn(p[j].y - pt.y);
				if (c > 0 && u < 0 && v >= 0) num++;
				if (c < 0 && u >= 0 && v < 0) num--;
			}
			return num != 0;
		}

		template<typename T>
		double polygon_perimeter(const Polygon<T>& p) {
			/**
			* 计算多边形的周长
			*/

			double ans = 0;
			int n = p.size();
			for (int i = 0; i < n; i++)
				ans += point_point_dist(p[i], p[(i + 1) % n]);
			return ans;
		}

		template<typename T>
		double polygon_area(const Polygon<T>& p) {
			/**
			* 计算多边形的面积
			* 注意面积存在正负：
			*	逆时针遍历点算出来就是正的
			*	顺时针遍历点算出来就是负的
			*/

			double area = 0;
			int n = p.size();
			for (int i = 0; i < n; i++)
				area += cross(p[i], p[(i + 1) % n]);
			return abs(area) / 2.0;
		}

		template<typename T>
		Point<T> polygon_center_point(const Polygon<T>& p) {
			/**
			* 求多边形的重心
			*/

			Point<T> ans(0, 0);
			int n = p.size();
			if (polygon_area(p, n) == 0) return ans;
			for (int i = 0; i < n; i++)
				ans = ans + (p[i] + p[(i + 1) % n]) * cross(p[i], p[(i + 1) % n]);
			return ans / polygon_area(p, n) / 6;
		}

		/* 补充知识：
				凸多边形：是指所有内角大小都在 [0, 180] 范围内的简单多边形
				凸包：在平面上能包含所有给定点的最小凸多边形叫做凸包
		*/
		template<typename T>
		Polygon<T> convex_hull(vector<Point<T>> p) {
			/**
			* 求凸包，凸包顶点放在 ch 中
			* Andrew 法：
			*	先对所有点排序
			*	求上下凸包 (查每个边相较于上一条边的拐弯方向)
			*	然后合并
			*	最后得到的点是逆时针顺序的
			*/

			Polygon<T> ch;

			if (p.size() == 0 or p.size() == 1 or p.size() == 2) {
				for (var& i : p) ch.pb(i);
				return ch;
			}

			int n = p.size();
			n = unique(p.begin(), p.end()) - p.begin(); // 去除重复点    
			ch.resize(2 * n);
			sort(p.begin(), p.end());  // 对点排序：按 x 从小到大排序，如果 x 相同，按 y 排序
			int v = 0;

			// 求下凸包，如果 p[i] 是右拐弯的，这个点不在凸包上，往回退
			for (int i = 0; i < n; i++) {
				while (v > 1 && sgn(cross(ch[v - 1] - ch[v - 2], p[i] - ch[v - 1])) <= 0)
					v--;
				ch[v++] = p[i];
			}

			// 求上凸包
			for (int i = n - 1, j = v; i >= 0; i--) {
				while (v > j && sgn(cross(ch[v - 1] - ch[v - 2], p[i] - ch[v - 1])) <= 0)
					v--;
				ch[v++] = p[i];
			}

			ch.resize(v - 1);
			return ch;
		}

		template<typename T>
		int point_circle_relation(const Point<T>& p, const Circle<T>& C) {
			/**
			* 点和圆的关系 (根据点到圆心的距离判断)
			* 返回值：
			*	0: 点在圆内
			*	1: 点在圆上
			*	2: 点在圆外
			*/

			double dst = point_point_dist(p, C.c);
			if (sgn(dst - C.r) < 0) return 0;
			if (sgn(dst - C.r) == 0) return 1;
			return 2;
		}

		template<typename T>
		int line_circle_relation(const Line<T>& v, const Circle<T>& C) {
			/**
			* 直线和圆的关系 (根据圆心到直线的距离判断)
			* 返回值：
			*	0: 直线和圆相交
			*	1: 直线和圆相切
			*	2: 直线在圆外
			*/

			double dst = point_line_dis(C.c, v);
			if (sgn(dst - C.r) < 0) return 0;
			if (sgn(dst - C.r) == 0) return 1;
			return 2;
		}

		template<typename T>
		int segment_circle_relation(const Segment<T> v, const Circle<T> C) {
			/**
			* 线段和圆的关系 (根据圆心到线段的距离判断)
			* 返回值：
			*	0: 线段在圆内
			*	1: 线段和圆相切
			*	2: 线段在圆外
			*/

			double dst = point_segment_dis(C.c, v);
			if (sgn(dst - C.r) < 0) return 0;
			if (sgn(dst - C.r) == 0) return 1;
			return 2;
		}

		template<typename T>
		PointSet<T> line_cross_circle_points(const Line<T>& v, const Circle<T>& C) {
			/**
			* 求直线和圆的交点
			* 传入直线 v、圆 C
			* 返回值：交点集合
			*/

			PointSet<T> se;
			if (line_circle_relation(v, C) == 2)  return se;   // 无交点
			Point<T> q = point_line_proj(C.c, v);              // 圆心在直线上的投影点
			double d = point_line_dis(C.c, v);                 // 圆心到直线的距离
			double k = sqrt(C.r * C.r - d * d);
			if (sgn(k) == 0) {                                 // 1个交点，直线和圆相切
				se.push_back(q);
				return se;
			}
			Point<T> n = (v.p2 - v.p1) / vector_len(v.p2 - v.p1);     // 单位向量
			se.push_back(q + n * k);
			se.push_back(q - n * k);
			return se;                                         // 2个交点
		}

		template<typename T>
		PointSet<T> line_polygon_cross_points(const Line<T>& l, const Polygon<T>& p) {
			/**
			* 求直线和多边形的交点
			* 传入直线 v、多边形 p
			* 返回值：交点集合
			*/

			set<Point<T>> se;
			int n = p.size();
			for (int i = 0; i < n; i++) {
				Line<T> nl(p[i], p[(i + 1) % n]);
				int status = line_line_relation(l, nl);
				if (status == 1) {
					// 重合
					se.insert(p[i]);
					se.insert(p[(i + 1) % n]);
				}
				else if (status == 2) {
					// 相交
					Point<T> point = line_line_cross_point(l, nl);
					if (point_segment_relation(point, nl)) {
						se.insert(point);
					}
				}
			}
			PointSet<T> ve;
			for (auto& i : se)ve.push_back(i);
			return ve;
		}

		template<typename T>
		double circle_arc_area(const Circle<T>& c, const db& angle) {
			/**
			* 计算扇形面积
			* 角度传入角度制
			*/
			return c.area() * angle % 360 / 360.0;
		}

		template<typename T>
		double circle_area(const Circle<T>& c) {
			/**
			* 计算圆面积
			*/
			return c.area();
		}

		template<typename T>
		void angle_polar_sort_atan2(vector<Point<T>>& points, const Point<T>& reference = Point<T>(0, 0)) {
			/**
			* atan2 极角排序 (逆时针排序)
			* 传入点集 points 和参考点 reference
			*/
			sort(points.begin(), points.end(),
				[&](const Point<T>& a, const Point<T>& b)->bool
				{ return a.polar_angle(reference) < b.polar_angle(reference); });
		}

		template<typename T>
		void angle_polar_sort_cross(vector<Point<T>>& points, const Point<T>& reference = Point<T>(0, 0)) {
			/**
			* cross 极角排序 (逆时针排序)
			* 传入点集 points 和参考点 reference
			*/
			sort(points.begin(), points.end(),
				[&](Point<T> a, Point<T> b)->bool {
					a = a - reference; b = b - reference;
					if (a.quadrant() != b.quadrant())return a.quadrant() < b.quadrant();
					return sgn(cross(a, b)) > 0; // b 在 a 逆时针方向
				});
		}

		template<typename T>
		db farthest_point_to_point_dis(const Polygon<T>& p, Point<T>& a, Point<T>& b) {
			/**
			* 旋转卡壳求最远点对及其距离
			* p 为凸包，ab 为最远点对
			* 返回值为最远点对的距离
			*/

			db mx = 0;
			int n = p.size();
			// 经典单调性，双指针枚举即可
			int j = 1;
			for (int i = 0; i < n; i++) { // 实际在枚举边
				// 利用面积判断离当前边最远的点
				while (abs(area_parallelogram(p[(i + 1) % n], p[i], p[(j + 1) % n])) >
					abs(area_parallelogram(p[(i + 1) % n], p[i], p[j]))) {
					j = (j + 1) % n;
				}
				db d1 = point_point_dist(p[i], p[j]);
				if (d1 > mx) {
					mx = d1;
					a = p[i];
					b = p[j];
				}
				db d2 = point_point_dist(p[(i + 1) % n], p[j]);
				if (d2 > mx) {
					mx = d2;
					a = p[(i + 1) % n];
					b = p[j];
				}
			}
			return mx;
		}


		template<typename T>
		long long farthest_point_to_point_dis2(const Polygon<T>& p, Point<T>& a, Point<T>& b) {
			/**
			* 旋转卡壳求最远点对及其距离
			* p 为凸包，ab 为最远点对
			* 返回值为最远点对的距离
			*/

			long long mx = 0;
			int n = p.size();
			// 经典单调性，双指针枚举即可
			int j = 1;
			for (int i = 0; i < n; i++) { // 实际在枚举边
				// 利用面积判断离当前边最远的点
				while (abs(area_parallelogram(p[(i + 1) % n], p[i], p[(j + 1) % n])) >
					abs(area_parallelogram(p[(i + 1) % n], p[i], p[j]))) {
					j = (j + 1) % n;
				}
				long long d1 = point_point_dist2(p[i], p[j]);
				if (d1 > mx) {
					mx = d1;
					a = p[i];
					b = p[j];
				}
				long long d2 = point_point_dist2(p[(i + 1) % n], p[j]);
				if (d2 > mx) {
					mx = d2;
					a = p[(i + 1) % n];
					b = p[j];
				}
			}
			return mx;
		}

		template<typename T>
		db farthest_point_to_point_dis(const Polygon<T>& p) {
			/**
			* 旋转卡壳求最远点对及其距离
			* p 为凸包，ab 为最远点对
			* 返回值为最远点对的距离
			*/

			db mx = 0;
			int n = p.size();
			// 经典单调性，双指针枚举即可
			int j = 1;
			for (int i = 0; i < n; i++) { // 实际在枚举边
				// 利用面积判断离当前边最远的点
				while (abs(area_parallelogram(p[(i + 1) % n], p[i], p[(j + 1) % n])) >
					abs(area_parallelogram(p[(i + 1) % n], p[i], p[j]))) {
					j = (j + 1) % n;
				}
				db d1 = point_point_dist(p[i], p[j]);
				db d2 = point_point_dist(p[(i + 1) % n], p[j]);
				mx = max({ d1, d2, mx });
			}
			return mx;
		}


		template<typename T>
		long long farthest_point_to_point_dis2(const Polygon<T>& p) {
			/**
			* 旋转卡壳求最远点对及其距离
			* p 为凸包，ab 为最远点对
			* 返回值为最远点对的距离
			*/

			long long mx = 0;
			int n = p.size();
			// 经典单调性，双指针枚举即可
			int j = 1;
			for (int i = 0; i < n; i++) { // 实际在枚举边
				// 利用面积判断离当前边最远的点
				while (abs(area_parallelogram(p[(i + 1) % n], p[i], p[(j + 1) % n])) >
					abs(area_parallelogram(p[(i + 1) % n], p[i], p[j]))) {
					j = (j + 1) % n;
				}
				long long d1 = point_point_dist2(p[i], p[j]);
				long long d2 = point_point_dist2(p[(i + 1) % n], p[j]);
				mx = max({ d1, d2, mx });
			}
			return mx;
		}

		template<typename T>
		class Point {
		private:
			int id;

		public:
			T x, y;

			Point(T x = 0, T y = 0) : x(x), y(y), id(0) {}

			double polar_angle(const Point<T>& reference = Point(0, 0)) const {
				/**
				* 计算 this 点相较于 reference 点的极角
				*/
				double res = atan2(y - reference.y, x - reference.x);
				if (sgn(res) < 0) res += (2 * PI);
				return res;
			}

			double len() const { return sqrt((*this) * (*this)); } // 向量长度
			T len2() const { return (*this) * (*this); } // 向量长度的平方

			int quadrant() {
				/**
				* 求点所在象限，包括了 xy 正负半轴
				* 可用于叉积法 (cross) 极角排序
				*/
				if (x > 0 && y >= 0)return 1;      // 包含了 y 非负半轴
				else if (x <= 0 && y > 0)return 2; // 包含了 x 非正半轴
				else if (x < 0 && y <= 0)return 3; // 包含了 y 非正半轴
				else if (x >= 0 && y < 0)return 4; // 包含了 x 非负半轴
				else return 0; // 原点
			}

			void set_id(int id) { this->id = id; }
			int get_id()const { return id; }

			Point operator- (const Point& B) const { return Point(x - B.x, y - B.y); }
			Point operator+ (const Point& B) const { return Point(x + B.x, y + B.y); }
			T operator^ (const Point<T>& B) const { return x * B.y - y * B.x; } // 叉积
			T operator* (const Point<T>& B) const { return x * B.x + y * B.y; } // 点积
			Point operator* (const T& B) const { return Point(x * B, y * B); }
			Point operator/ (const T& B) const { return Point(x / B, y / B); }
			bool operator< (const Point& B) const { return cmp(x, B.x) == -1 || (cmp(x, B.x) == 0 && cmp(y, B.y) == -1); }
			bool operator> (const Point& B) const { return cmp(x, B.x) == 1 || (cmp(x, B.x) == 0 && cmp(y, B.y) == 1); }
			bool operator== (const Point& B) const { return cmp(x, B.x) == 0 && cmp(y, B.y) == 0; }
			bool operator!= (const Point& B) const { return !(*this == B); }

			friend ostream& operator<<(ostream& out, const Point& a) {
				out << '(' << a.x << ", " << a.y << ')';
				return out;
			}
		};

		template<typename T>
		class Line {
		public:
			Point<T> p1, p2; // 线上的两个点
			Line() {}
			Line(const Point<T>& p1, const Point<T>& p2) :p1(p1), p2(p2) {}
			Line(const Point<T>& p, double angle) {
				/**
				* 根据一个点和倾斜角 angle 确定直线，0 <= angle < pi
				*/
				p1 = p;
				if (sgn(angle - PI / 2) == 0) { p2 = (p1 + Point<T>(0, 1)); }
				else { p2 = (p1 + Point<T>(1, tan(angle))); }
			}
			Line(double a, double b, double c) {     // ax + by + c = 0
				if (sgn(a) == 0) {
					p1 = Point<T>(0, -c / b);
					p2 = Point<T>(1, -c / b);
				}
				else if (sgn(b) == 0) {
					p1 = Point<T>(-c / a, 0);
					p2 = Point<T>(-c / a, 1);
				}
				else {
					p1 = Point<T>(0, -c / b);
					p2 = Point<T>(1, (-c - a) / b);
				}
			}
			friend ostream& operator<<(ostream& out, const Line<T>& a) {
				out << "[" << a.p1 << ", " << a.p2 << "]";
				return out;
			}

			bool is_no_k() {
				return sgn(p2.x - p1.x) == 0;
			}

			double k() const {
				/**
				* 计算斜率 k
				* 注意：直线平行于 y 轴，斜率 k 不存在
				*/
				assert(sgn(p2.x - p1.x) != 0); // 垂直线，斜率不存在
				return double(p2.y - p1.y) / (p2.x - p1.x);
			}

			double b() const {
				/**
				* 计算截距 b
				* 注意：直线平行于 y 轴，斜率 k 不存在
				*/
				assert(sgn(p2.x - p1.x) != 0); // 垂直线，斜率不存在
				double _k = k();
				return p1.y - _k * p1.x;
			}
		};

		template<typename T>
		class Polygon : public vector<Point<T>> {
		public:
			Polygon() {}
			Polygon(int n) :vector<Point<T>>(n) {}

			// 多边形的周长
			T perimeter() {
				T ans = 0;
				int n = this->size();
				for (int i = 0; i < n; i++)
					ans += point_point_dist((*this)[i], (*this)[(i + 1) % n]);
				return ans;
			}

			// 多边形的面积
			db area() {
				T area = 0;
				int n = this->size();
				for (int i = 0; i < n; i++) {
					area += cross((*this)[i], (*this)[(i + 1) % n]);
				}
				return abs(area) / 2.0;
			}

			// 多边形的面积 * 2
			long long area2() {
				long long area = 0;
				int n = this->size();
				for (int i = 0; i < n; i++) {
					area += cross((*this)[i], (*this)[(i + 1) % n]);
				}
				return abs(area);
			}

			// 多边形的面积
			db area_directed() {
				T area = 0;
				int n = this->size();
				for (int i = 0; i < n; i++) {
					area += cross((*this)[i], (*this)[(i + 1) % n]);
				}
				return area / 2.0;
			}

			// 多边形的面积 * 2
			long long area2_directed() {
				long long area = 0;
				int n = this->size();
				for (int i = 0; i < n; i++) {
					area += cross((*this)[i], (*this)[(i + 1) % n]);
				}
				return area;
			}

			bool winding_order() {
				/**
				* 检查多边形里的点按什么顺序排列
				* 根据多边形有向面积判断
				* 返回值：
				*	0：顺时针
				*	1：逆时针
				*/
				return (area2_directed() > 0);
			}

			// atan2 极角排序，默认逆时针排序
			void polar_angle_sort_atan2(const Point<T>& reference = Point<T>(0, 0)) {
				sort(this->begin(), this->end(),
					[&](const Point<T>& a, const Point<T>& b)->bool
					{ return a.polar_angle(reference) < b.polar_angle(reference); });
			}

			// cross 极角排序，默认逆时针排序
			void polar_angle_sort_cross(const Point<T>& reference = Point<T>(0, 0)) {
				sort(this->begin(), this->end(),
					[&](Point<T> a, Point<T> b)->bool {
						a = a - reference; b = b - reference;
						if (a.quadrant() != b.quadrant())return a.quadrant() < b.quadrant();
						return sgn(cross(a, b)) > 0;
					});
			}

			friend ostream& operator<<(ostream& out, const Polygon<T>& a) {
				out << "[";
				for (int i = 0; i < a.size(); i++)
					out << a[i] << ",]"[i == a.size() - 1];
				return out;
			}
		};

		template<typename T>
		class Circle {
		public:
			Point<T> c;  // 圆心
			T r;         // 半径
			Circle() {}
			Circle(Point<T> c, T r) :c(c), r(r) {}
			Circle(T x, T y, T _r) { c = Point<T>(x, y); r = _r; }
			double area() const { return PI * r * r; }
			double arc_area(const db& angle) const { return area() * angle / 360.0; }
			friend ostream& operator<<(ostream& out, const Circle<T>& a) {
				out << "(" << a.c << ", " << a.r << ")";
				return out;
			}
		};
	}


	template <typename T>
	class Trie01Vector {
	private:
		struct TrieNode {
			array<int, 2> children; // 指向孩子的 id
			int cnt; // 记录经过该节点的数字个数
			/* 可再添些变量，比如 区间查询问题用到的 边归属：int id； */
			/* 本板子可以进行区间询问，添加 边归属：int id；后，修改插入和询问函数即可。  */

			TrieNode() : children{ 0, 0 }, cnt(0) {}
		};

		vector<TrieNode> tr;
		int newNode() { tr.push_back(TrieNode()); return ++nodeCnt; }

		int MAX; 	      // MAX：数的最大二进制位数
		int nodeCnt = 1;
		int root = 1;

	public:
		Trie01Vector(int MAX) :MAX(MAX), tr(2) {}

		void insert(T num) {
			int now = root;
			tr[now].cnt++;

			// 枚举二进制数位
			for (int i = MAX; i >= 0; i--) {
				bool bit = (num >> i) & 1; // 当前数位
				if (not tr[now].children[bit]) {
					// 没有对应数位的边，则自己造边
					int tmp = newNode();
					tr[now].children[bit] = tmp;
				}
				now = tr[now].children[bit];
				tr[now].cnt++;
			}
		}

		// 看看 num 跟 01 trie 里的哪个元素 XOR 最后的值最大
		T find_max_xor(T num) {
			int now = root;
			T maxXor = 0;
			for (int i = MAX; i >= 0; i--) {
				bool bit = (num >> i) & 1;
				// 尽量选与自己相反的，从而进 1
				if (tr[now].children[!bit]) {
					maxXor |= (1ll << i);
					now = tr[now].children[!bit];
				}
				else {
					// 没有的话只能乖乖往下走，这一位变 0
					now = tr[now].children[bit];
				}
			}
			return maxXor;
		}

		// 看看 num 跟 01 trie 里的哪个元素 XOR 最后的值最小
		T find_min_xor(T num) {
			auto now = root;
			T minXor = 0;
			for (int i = MAX; i >= 0; i--) {
				bool bit = (num >> i) & 1;
				if (tr[now].children[bit]) {
					now = tr[now].children[bit];
				}
				else {
					minXor |= (1ll << i);
					now = tr[now].children[bit];
				}
			}
			return minXor;
		}

		void erase(T num) {
			int now = root;
			if (tr[now].cnt <= 0) return;
			tr[now].cnt--;

			for (int i = MAX; i >= 0; i--) {
				bool bit = (num >> i) & 1;
				if (tr[now].children[bit] == 0) return;

				int nxt = tr[now].children[bit];
				tr[nxt].cnt--;

				if (tr[nxt].cnt == 0) {
					// 注意：这里实际没有从vector中删除节点，只是标记为未使用
					tr[now].children[bit] = 0;
				}

				now = nxt;
			}
		}

		// 查询数字是否存在
		bool contains(T num) {
			int now = root;
			for (int i = MAX; i >= 0; i--) {
				bool bit = (num >> i) & 1;
				if (tr[now].children[bit] == 0 || tr[tr[now].children[bit]].cnt <= 0) {
					return false;
				}
				now = tr[now].children[bit];
			}
			return true;
		}

		// 获取当前Trie中的数字个数
		int size() {
			return tr[root].cnt;
		}
	};

	template <typename T, size_t N>
	class Trie01Array {
	private:
		struct TrieNode {
			array<int, 2> children; // 指向孩子的 id
			/* 可再添些变量，比如 区间查询问题用到的 边归属：int id； */
			/* 本板子可以进行区间询问，添加 边归属：int id；后，修改插入和询问函数即可。  */
		};

		// N = n * MAX, n 个数 * MAX 最大二进制位数
		unique_ptr<array<TrieNode, N>> t;
		array<TrieNode, N>& tr = *t;

		int newNode() { return ++nodeCnt; }

		int MAX; 	      // MAX：数的最大二进制位数
		int root = 1;
		int nodeCnt = 1;
	public:
		Trie01Array(int MAX) :MAX(MAX), t(make_unique<array<TrieNode, N>>()) {}

		void insert(T num) {
			int now = root;
			// 枚举二进制数位
			for (int i = MAX; i >= 0; i--) {
				bool bit = (num >> i) & 1; // 当前数位
				if (not tr[now].children[bit]) {
					// 没有对应数位的边，则自己造边
					tr[now].children[bit] = newNode();
				}
				now = tr[now].children[bit];
			}
		}

		// 看看 num 跟 01 trie 里的哪个元素 XOR 最后的值最大
		T find_max_xor(T num) {
			int now = root;
			T maxXor = 0;
			for (int i = MAX; i >= 0; i--) {
				bool bit = (num >> i) & 1;
				// 尽量选与自己相反的，从而进 1
				if (tr[now].children[!bit]) {
					maxXor |= (1ll << i);
					now = tr[now].children[!bit];
				}
				else {
					// 没有的话只能乖乖往下走，这一位变 0
					now = tr[now].children[bit];
				}
			}
			return maxXor;
		}

		// 看看 num 跟 01 trie 里的哪个元素 XOR 最后的值最小
		T find_min_xor(T num) {
			int now = root;
			T minXor = 0;
			for (int i = MAX; i >= 0; i--) {
				bool bit = (num >> i) & 1;
				if (tr[now].children[bit]) {
					now = tr[now].children[bit];
				}
				else {
					minXor |= (1ll << i);
					now = tr[now].children[bit];
				}
			}
			return minXor;
		}
	};

	struct TrieNode {
		unordered_map<char, TrieNode*> children;
		int id;
		bool is_end;
		TrieNode() : is_end(false), id(0) {}
		TrieNode(int id) : is_end(false), id(id) {}
	};

	class Trie {
	private:
		int nodeCnt = 0;

		TrieNode* newNode() {
			nodeCnt++;
			ends.push_back(0);
			prefixs.push_back(0);
			return new TrieNode(nodeCnt);
		}

	public:
		TrieNode* root;

		vector<int> ends;     // 在 id 点截止的字符串个数
		vector<int> prefixs;  // 经过 id 点的字符串个数 (共同前缀个数)

		Trie() : ends(1), prefixs(1) {
			root = newNode();
		}

		void insert(const string& word) {
			int id = 1;
			prefixs[id]++;

			TrieNode* now = root;
			for (const char& c : word) {

				// 没找到当前字母就建新边
				if (not now->children.count(c))
					now->children[c] = newNode();

				now = now->children[c];
				id = now->id;
				prefixs[id]++;
			}

			now->is_end = true;
			ends[id]++;
		}

		// 查字典树中有多少带 pre 前缀的字符串
		int startsWith(const string& pre) {
			TrieNode* node = root;
			for (const char& c : pre) {
				if (not node->children.count(c)) return 0;
				node = node->children[c];
			}
			return prefixs[node->id];
		}

		// 查字典树中有几个 word
		int search(const string& word) {
			TrieNode* node = root;
			for (const char& c : word) {
				if (not node->children.count(c)) return 0;
				node = node->children[c];
			}
			return ends[node->id];
		}
	};

	class Dsu {
	private:
		function<void(int, int, Dsu&)>uoion_func;
	public:
		int n;
		vector<int>root;
		Dsu(int n, function<void(int, int, Dsu&)>func =
			[](int x, int y, Dsu& d)->void { // 默认合并函数
				var fx = d.find(x), fy = d.find(y);
				if (fx == fy)return;
				d.root[fy] = fx;
			}
		) :n(n), root(n)
		{
			fa(i, 0, n - 1)
				root[i] = i;
		}

		int find(int x) {
			if (root[x] != x)return root[x] = find(root[x]);
			return root[x];
		}

		// 前者为根，后者为子
		void uoion(int x, int y) {
			uoion_func(x, y, *this);
		}
	};

	tuple<int, vector<vector<int>>, vector<ll>, vector<int>> scc_shrink(int n, const vector<vector<int>>& g, const vector<ll>& a) {
		/**
		* 缩点板子，找全联通分量
		* 传入：原图节点个数 n、原图 g、原图节点权值 a
		* 返回：新图节点个数 scc、新图 ng (注意：没有去重)、新图节点权值 na
		* 
		* - 缩点完毕后变成有向无环图，可能需要拓扑排序
		* - 可能还要看一个缩点原来有哪些节点，具体看题面而定
		* 
		* 强连通：一张有向图的节点两两互相可达
		* 强连通分量：极大的强连通子图
		* Tarjan 算法：通过记录深搜遍历中每个节点的第一次访问时间来找到强连通分量的根以及其余节点
		*/

		auto dfn = vector<int>(n + 1); // dfs序时间戳
		auto low = vector<int>(n + 1); // i 点能回溯到的最顶端祖先的 dfn
		auto ins = vector<int>(n + 1); // 是否在栈内
		auto bel = vector<int>(n + 1); // i 点属于第几个强连通分量
		auto st = stack<int>();        // 正在处理的栈
		int time = 0;                  // 时间戳
		int scc = 0;                   // 强连通分量个数，同时也是新图节点个数(单个节点也是个 scc)

		auto tarjan = [&](auto tarjan, int now)->void {
			dfn[now] = low[now] = ++time;
			st.push(now);
			ins[now] = 1;
			for (const int& to : g[now]) {
				if (!dfn[to]) { // 未遍历到的节点
					tarjan(tarjan, to);
					low[now] = min(low[now], low[to]);
				}
				else if (ins[to]) { // 环
					low[now] = min(low[now], dfn[to]);
				}
			}
			// 弹栈缩点
			if (low[now] == dfn[now]) {
				scc++;
				while (1) {
					int cur = st.top();
					st.pop();
					ins[cur] = 0;
					bel[cur] = scc;
					if (now == cur)break;
				}
			}
			};
		for (int i = 1; i <= n; i++) if (!dfn[i]) tarjan(tarjan, i);


		// 构建新有向图
		auto ng = vector<vector<int>>(scc + 1); // 缩点后的新图
		auto na = vector<ll>(scc + 1);          // 缩点后的新权值
		auto siz = vector<int>(scc + 1);        // 缩点后新节点包含了多少原节点(可选)
		for (int now = 1; now <= n; now++) {
			for (const int& to : g[now])
				if (bel[to] != bel[now]) // 不在同一个 scc 内
					ng[bel[now]].push_back(bel[to]);
			na[bel[now]] += a[now];
			siz[bel[now]]++;
		}

		return make_tuple(scc, ng, na, siz);
	}

	template<typename T> 
	T randint(T l, T r) {
		// 随机整数
		static std::mt19937_64 gen(std::random_device{}()); 
		std::uniform_int_distribution<T> dist(l, r); 
		return dist(gen); 
	}

	struct kruskal_edge {
		ll u, v, w;
	};
	tuple<vector<vector<int>>, vector<ll>, vector<vector<int>>> kruskal_rebuildTree(int n, vector<kruskal_edge>& e) {
		/**
		* 克鲁斯卡尔重构树
		* 是个二叉树，叶子节点是原图节点，非叶子节点是原图的边
		* 传入：原图节点个数 n、边数组 e
		* 返回：新图 g、新节点权值数组 val、倍增父节点数组 p
		*
		* - 查两点间最小瓶颈路问题：求从 u 到 v 的所有路径中，最大边权最小的那条路径的最大边权值，答案就是 LCA(u, v) 的点权。
		* - “边权限制”下的连通性查询：查询仅经过边权不超过 k 的边，两点 u 和 v 能否连通。
		*/

		// 并查集
		int nn = 2 * n - 1; // 重构树一共 2n - 1 个节点
		auto s = vector<int>(nn + 1);
		for (int i = 1; i <= nn; i++)s[i] = i;
		auto find = [&](auto find, int x)->int {
			if (s[x] == x)return x;
			return s[x] = find(find, s[x]);
			};
		sort(e.begin() + 1, e.end(), [](const kruskal_edge& a, const kruskal_edge& b)->bool {
			return a.w < b.w;
			});

		// 克鲁斯卡尔
		auto g = vector<vector<int>>(nn + 1);
		auto val = vector<ll>(nn + 1); // 新开节点(原图的边)的权值
		auto p = vector<vector<int>>(nn + 1, vector<int>(30)); // 倍增父节点
		int now = n;
		fa(i, 1, (int)e.size() - 1) {
			const auto& [u, v, w] = e[i];
			int e1 = find(find, u);
			int e2 = find(find, v);
			if (e1 != e2) {
				val[now] = w;
				g[++now].push_back(e1), g[now].push_back(e2); // 建二叉树
				s[e1] = now, s[e2] = now;
				p[e1][0] = now, p[e2][0] = now; // 倍增父节点初始化
				// 此处可能还需要进行倍增DP初始化：dp[e1/e2][0] = f(val[now]);
				// 以及其它一维数组信息收集，如：sum[now] = sum[e1] + sum[e2];
			}
		}
		return { g,val,p };
	}

	vector<int> cut_point(int n, const vector<vector<int>>& g) {
		// 割点：在一个连通无向图中，如果某个顶点以及与其相关联的所有边，图的连通分量数量增加，那么这个顶点就是一个割点。

		auto dfn = vector<int>(n + 1);  // dfs序时间戳
		auto low = vector<int>(n + 1);  // i 点能回溯到的最顶端祖先的 dfn，无向图要考虑能否绕过直接前驱
		auto cnt = vector<bool>(n + 1); // 是否是割点
		int time = 0;                   // 时间戳

		auto tarjan = [&](auto tarjan, int now, int root)->void {
			dfn[now] = low[now] = ++time;
			int child = 0; // root 的子节点
			for (const int& to : g[now]) {
				if (!dfn[to]) {
					tarjan(tarjan, to, root);
					low[now] = min(low[now], low[to]);
					if (low[to] >= dfn[now] and now != root)
						// to 无法绕过 now 达到 now 的祖先
						cnt[now] = 1;
					if (now == root) child++;
				}
				low[now] = min(low[now], dfn[to]);
			}
			// 因为考虑到子节点的子节点能回来的特性，这里的子节点不是单纯的 root 子节点
			// 子节点的子节点绕不回 root 节点的个数 child >= 2 时产生割点
			if (child >= 2 and now == root)cnt[root] = 1;
			};
		for (int i = 1; i <= n; i++) if (!dfn[i]) tarjan(tarjan, i, i);

		auto ans = vector<int>();
		for (int i = 1; i <= n; i++)if (cnt[i])ans.push_back(i);
		return ans;
	}

	vector<pair<int, int>> cut_edge(int n, const vector<vector<int>>& g) {
		// 割边（桥）：在一个连通无向图中，如果删除某条边后，图的连通分量数量增加，那么这条边就是一个割边（桥）。

		vector<int> dfn(n + 1);   // dfs序时间戳
		vector<int> low(n + 1);   // 能回溯到的最早祖先的dfn
		vector<pair<int, int>> bridges; 
		int time = 0;

		auto tarjan = [&](auto self, int u, int parent)->void {
			dfn[u] = low[u] = ++time;
			for (int v : g[u]) {
				if (v == parent) continue;

				if (!dfn[v]) {
					self(self, v, u);
					low[u] = min(low[u], low[v]);
					// 判断桥：如果v无法通过其他边回溯到u或更早，则(u,v)是桥
					if (low[v] > dfn[u]) {
						bridges.push_back({ min(u, v), max(u, v) }); // 按小节点在前存储，避免重复
					}
				}
				else low[u] = min(low[u], dfn[v]);
			}
			};

		for (int i = 1; i <= n; i++) if (!dfn[i]) tarjan(tarjan, i, 0);
		return bridges;
	}

	vector<vector<int>> bcc_point(int n, const vector<vector<int>>& g) {
		/**
		* 无向图点双连通分量：不包含割点的极大连通子图
		* 
		* - 可能还要进一步用并查集缩点，然后建树
		*/

		vector<int> dfn(n + 1);   // dfs序时间戳
		vector<int> low(n + 1);   // 能回溯到的最早祖先的dfn
		vector<bool> cut(n + 1, false);
		stack<int> stk;
		vector<vector<int>> bccs; // 存储所有点双连通分量
		int time = 0;
		int root = 0;

		auto tarjan = [&](auto self, int u, int fa) -> void {
			dfn[u] = low[u] = ++time;
			stk.push(u);

			// 根节点的特殊情况
			if (u == root && g[u].empty()) {
				bccs.push_back({ u });
				return;
			}

			int child = 0;
			for (int v : g[u]) {
				if (v == fa) continue;

				if (!dfn[v]) {
					self(self, v, u);
					low[u] = min(low[u], low[v]);

					if (low[v] >= dfn[u]) {
						child++;
						if (u != root || child > 1) {
							cut[u] = true;
						}

						// 发现一个点双连通分量
						vector<int> bcc;
						while (true) {
							int top = stk.top();
							stk.pop();
							bcc.push_back(top);
							if (top == v) break;
						}
						bcc.push_back(u);
						bccs.push_back(bcc);
					}
				}
				else {
					low[u] = min(low[u], dfn[v]);
				}
			}
			};

		for (int i = 1; i <= n; i++) {
			if (!dfn[i]) {
				root = i;
				tarjan(tarjan, i, 0);

				// 处理根节点可能剩余的点
				if (!stk.empty()) {
					vector<int> bcc;
					while (!stk.empty()) {
						bcc.push_back(stk.top());
						stk.pop();
					}
					bccs.push_back(bcc);
				}
			}
		}

		return bccs;
	}

	vector<vector<int>> bcc_edge(int n, const vector<vector<int>>& g) {
		/**
		* 无向图边双连通分量：不包含桥的极大连通子图
		*
		* - 可能还要进一步用并查集缩点，然后建树
		*/

		vector<int> dfn(n + 1);   // dfs序时间戳
		vector<int> low(n + 1);   // 能回溯到的最早祖先的dfn
		vector<int> bcc_id(n + 1, 0); // 每个点所属的边双连通分量id
		stack<int> stk;
		vector<vector<int>> bccs; // 存储所有边双连通分量
		int time = 0;
		int bcc_cnt = 0;

		auto tarjan = [&](auto self, int u, int fa) -> void {
			dfn[u] = low[u] = ++time;
			stk.push(u);

			for (int v : g[u]) {
				if (v == fa) continue;

				if (!dfn[v]) {
					self(self, v, u);
					low[u] = min(low[u], low[v]);
				}
				else {
					low[u] = min(low[u], dfn[v]);
				}
			}

			// 如果u是边双连通分量的根
			if (low[u] == dfn[u]) {
				bcc_cnt++;
				vector<int> bcc;
				while (true) {
					int top = stk.top();
					stk.pop();
					bcc_id[top] = bcc_cnt;
					bcc.push_back(top);
					if (top == u) break;
				}
				bccs.push_back(bcc);
			}
			};

		for (int i = 1; i <= n; i++) {
			if (!dfn[i]) {
				tarjan(tarjan, i, 0);
			}
		}

		/* 缩点后建树参考代码，记得留意是否需要去重
			int nn = bccs.size();
			var s = vector<int>(n + 1);
			fa(i, 0, nn - 1) {
				for (int x : bccs[i])
					s[x] = i + 1;
			}

			var ng = vector<vector<pll>>(nn + 1); // 缩点后的树
			for (const var& [u, v, w] : e) {
				if (s[u] != s[v]) { // 是桥
					ng[s[u]].pb({ s[v],w });
					ng[s[v]].pb({ s[u],w });
				}
			}

			var dfs = [&](var dfs, int now, int fa)->void {
				for (const var& [to, w] : ng[now]) {
					if (to == fa)continue;
					dfs(dfs, to, now);
					siz[now] += siz[to] + 1;
				}
				};
			dfs(dfs, 1, 0);
		*/

		return bccs;
	}

	bool is_bipartite_graph(int n, const vector<vector<int>>& g) {
		/**
		* 染色法判断是否是二分图
		*
		* - 二分图一定不含奇环，可用做奇环检测
		*/

		var color = vector<short>(n + 1, -1);

		auto bfs_check = [&](int st)->bool {
			queue<int> q;
			q.push(st);
			color[st] = 0;

			while (!q.empty()) {
				int u = q.front();
				q.pop();

				for (int v : g[u]) {
					if (color[v] == -1) {
						color[v] = color[u] ^ 1;
						q.push(v);
					}
					else if (color[v] == color[u]) {
						return false;
					}
				}
			}
			return true;
			};

		fa(i, 1, n)
			if (color[i] == -1 and !bfs_check(i))
				return false;

		return true;
	}

	tuple<
		int,         // 终点(最远点)
		ll,          // 最远距离
		vector<pii>  // 直径路径
	>
		diameter_of_tree(
			int n,                        // 原图节点个数
			const vector<vector<pll>>& g, // 原图
			int f,                        // 起点
			const bool& need_path = 0     // 是否需要直径路径
		)
	{
		/**
		* 树的直径
		* 
		* 性质：  
		*	1. 直径的两端点一定是叶子节点。  
		*
		*	2. 对于树中任意一点，距离它最远的点一定是某条直径的一个端点。
		* 
		*	3. 若一棵树存在多条直径，则这些直径必定相交于一点，且该交点是所有直径的公共中点。
		*
		*	4. 若有两棵树，第一棵树的直径端点为 (u, v)，第二棵树的直径端点为 (x, y)，用一条边将两棵树连接后，新树的直径端点必然在 ({u, v, x, y}) 中取两个点。  
		*	证明：
		*		若新直径不是原来任意一棵树的直径，则它必须跨越连接边。
		*		此时，新直径在每棵树内的部分一定是该树中距离连接点最远的点，而这样的点必然是原直径的端点。  
		*
		*	5. 在一棵树中，若在某个节点上接一个叶子节点，则至多改变原直径的一个端点。  
		*	证明：
		*		假设在节点 x 下接叶子节点 y，新直径变为 (u, y)，原直径为 (a, b)。
		*		那么有 dis(u, y) = dis(u, x) + 1 > dis(a, b)。
		*       若 dis(u, x) < dis(a, b)，则不等式不成立；
		*		若 dis(u, x) = dis(a, b)，则说明 (u, x) 已经是原树的一条直径，符合结论。  
		*/

		var path = vector<pii>();
		var suf = vector<int>();
		if (need_path)suf.resize(n + 1);

		var dfs = [&](var dfs, int now, int fa)->pll {
			var res = pll{ now, 1 }; // { 节点，now 到整个子树的最远距离 }
			for (const var& [to, w] : g[now]) {
				if (to == fa)continue;
				var tmp = dfs(dfs, to, now);
				tmp.py += w; // now 到 to 子树的最远距离
				if (tmp.py > res.py) {
					if (need_path)suf[now] = to;
					res = tmp;
				}
			}
			return res;
			};

		var res = dfs(dfs, f, 0);

		if (need_path) {
			int now = f;
			while (suf[now]) {
				path.pb({ now, suf[now] });
				now = suf[now];
			}
		}

		return { res.px, res.py, path };
	}

	int centroid_of_tree(int n, const vector<vector<int>>& g) {
		/**
		* 树的重心
		*
		* 性质：
		*	1. 树的重心如果不唯一，则至多有两个，且这两个重心相邻；
		*	   且删去它们的连边后，树将变为两个大小相同的连通分量。
		*
		*	2. 一个点是重心, 以这个点为根，它的每个子树的大小，都不会超过整个树大小的一半
		*
		*	3. 树中所有点到某个点的距离和中，到重心的距离和是最小的;
		*	   如果有两个重心，那么到它们的距离和一样。更进一步，距离和最小与是重心等价
		*
		*	4. 如果一个树增添或删去一个叶子，则整个树的同一个重心最多移动一个节点
		*
		*	5. 通过连接一条端点分别在两个树的边，来将两个树合并成一个，那么新的重心肯定是在原来这两个树的重心的路径上
		*
		*
		* 注意：若要多次求重心(点分治)，建议把函数内的都掏出来，防止多次刷新 siz/mxsiz 数组
		*/

		int root = 0;                   // 重心
		var siz = vector<int>(n + 1);   // i 子树大小
		var mxsiz = vector<int>(n + 1); // i 的每个子树大小的最大值

		var getroot = [&](var getroot, int now, int fa)->void {
			// 每次遍历整个树的时候都不用刷新 siz/mxsiz 数组，因为每次都会更新
			siz[now] = 1; mxsiz[now] = 0;

			for (int to : g[now]) {
				if (to == fa)continue;
				getroot(getroot, to, now);
				siz[now] += siz[to];                   // 计算 now 作为子树的大小
				mxsiz[now] = max(mxsiz[now], siz[to]); // 找 now 的每个子树大小的最大值
			}

			// 所有子树，包括父节点的
			mxsiz[now] = max(mxsiz[now], n - siz[now]);

			// 看看当前是不是重心
			// 要满足 now 的最大子树的大小不能超过 root 的最大子树的大小
			if (mxsiz[now] < mxsiz[root])root = now;
			};

		getroot(getroot, 1, 0);
		return root;
	}

	void centroid_decomposition(int n, vector<vector<int>>& g) {
		/**
		* 点分治
		* - 主要用于处理树上的路径问题(如：枚举计算两点对距离，即计算两点对到重心的距离)
		* - 核心思想是通过重心划分子连通块，从而将树平均分割成若干互不影响的子问题，复杂度为 O(nlogn)。
		*
		* 注意：本模板不完整，请自行补充或作为参考
		*/

		// 获取重心
		int root = 0;                   // 重心
		int now_n = n;                  // 当前子树的大小，因为重心一直会变，原重心会被删，所以得加上这个
		var siz = vector<int>(n + 1);   // i 的每个子树的大小
		var mxsiz = vector<int>(n + 1); // i 的每个子树的大小的最大值
		var vis = vector<bool>(n + 1);  // 存哪些点已经当过重心用了
		mxsiz[0] = 1e9;                 // 找重心用，随便设个最大值

		// 找重心
		var getroot = [&](var getroot, int now, int fa)->void {
			// 每次遍历整个树的时候都不用刷新 siz/mxsiz 数组，因为每次都会更新
			siz[now] = 1; mxsiz[now] = 0;

			for (int to : g[now]) {
				if (to == fa or vis[to])continue;
				getroot(getroot, to, now);
				siz[now] += siz[to];                   // 计算 now 作为子树的大小
				mxsiz[now] = max(mxsiz[now], siz[to]); // 找 now 的每个子树大小的最大值
			}

			// 所有子树，包括父节点的
			mxsiz[now] = max(mxsiz[now], now_n - siz[now]);

			// 看看当前是不是重心
			// 要满足 now 的最大子树的大小不能超过 root 的最大子树的大小
			if (mxsiz[now] < mxsiz[root])root = now;
			};

		var cal = [&](var cal, int now)->void {
			// 自行补充从重心 now 遍历子树，vis 到的不能再遍历

			};

		var dfs = [&](var dfs, int now)->void {
			vis[now] = 1;
			cal(cal, now); // now 是重心，以重心开始遍历整个子树

			// 对子树进行分治
			for (int to : g[now]) {
				if (vis[to])continue;
				now_n = siz[to]; // 分治子树，子树的大小，找重心用
				root = 0;        // 更新重心
				mxsiz[0] = 1e9;  // 更新重心用
				getroot(getroot, to, now); // 找重心
				dfs(dfs, to);
			}
			};

		getroot(getroot, 1, 0);
		dfs(dfs, root);

		return;
	}

	template<const int N>
	class Matrix {
		/**
		* 矩阵快速幂优化 dp
		* 
		*	已推出 dp 转移方程：f[i] = cal(f[i - 1])，该方程只在乎上一维；
		*	且 n 过大导致 O(n) 过不了时，考虑使用矩阵快速幂优化至 O(logn)。
		*
		*	将转移方程构造成矩阵连乘形式，即可套用快速幂：
		*		ans[i][1, n] = base[n, n] * ans[i - 1][1, n]
		*
		*	需要对 答案矩阵 ans 的第一项、系数矩阵 base 进行初始化
		*/

	public:
		array<array<ll, N + 1>, N + 1> mat;

		Matrix() {
			for (int i = 0; i <= N; i++)
				for (int j = 0; j <= N; j++)
					mat[i][j] = 0;
		}

		Matrix operator* (const Matrix& x) {
			// 矩阵乘法
			Matrix ret;
			for (int i = 0; i <= (1 << n) - 1; i++)
				for (int j = 0; j <= (1 << n) - 1; j++)
					for (int k = 0; k <= (1 << n) - 1; k++) {
						ret.mat[i][j] += mat[i][k] * x.mat[k][j];

						// 视情况可以进行修改，比如算和上一轮求和最大的:
						//		ret.mat[i][j] = max(ret.mat[i][j], mat[i][k] + x.mat[k][j]);
					}
			return ret;
		}

		Matrix ksm(ll k) {
			Matrix res, base;
			res.init_ans();
			base.init_base();

			while (k) {
				if (k & 1)res = res * base;
				base = base * base;
				k >>= 1;
			}
			return res;
		}

	private:
		void init_ans() {
			// 初始化第一项答案矩阵
			// 答案矩阵默认一维，全排在 [1][...]

		}

		void init_base() {
			// 初始化系数矩阵
			// 系数矩阵默认二维，注意 mat 赋值时可能需要 i j 互换 

		}
	};

	// 自定义哈希
	struct custom_hash {
		static uint64_t splitmix64(uint64_t x) {
			x += 0x9e3779b97f4a7c15;
			x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
			x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
			return x ^ (x >> 31);
		}

		// long long
		size_t operator()(uint64_t x) const {
			static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
			return splitmix64(x + FIXED_RANDOM);
		}

		// pair<long, long>
		size_t operator()(pair<uint64_t, uint64_t> x) const {
			static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
			return splitmix64(x.first + (x.second << 32) + FIXED_RANDOM);
		}

		//// 通用 pair 版本
		//template <typename T1, typename T2>
		//size_t operator()(const std::pair<T1, T2>& p) const {
		//	static const uint64_t FIXED_RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
		//	auto h1 = (*this)(p.first);
		//	auto h2 = (*this)(p.second);
		//	return splitmix64(h1 + splitmix64(h2 + FIXED_RANDOM));
		//}

		// 通用 tuple 版本
		template <typename T1, typename T2, typename T3>
		size_t operator()(const std::tuple<T1, T2, T3>& t) const {
			static const uint64_t FIXED_RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
			auto h1 = (*this)(std::get<0>(t));
			auto h2 = (*this)(std::get<1>(t));
			auto h3 = (*this)(std::get<2>(t));
			return splitmix64(h1 + splitmix64(h2 + splitmix64(h3 + FIXED_RANDOM)));
		}

		// string 版本
		size_t operator()(const std::string& s) const {
			static const uint64_t FIXED_RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
			uint64_t hash = FIXED_RANDOM;
			for (char c : s)hash = splitmix64(hash + static_cast<uint8_t>(c));
			return hash;
		}

		//// 通用版本，支持任意类型
		//template <typename T>
		//size_t operator()(const T& val) const {
		//	return (*this)(std::hash<T>{}(val));
		//}
	};
}
#endif