#ifndef LYNIA_H
#define LYNIA_H

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
					mp[i]++;
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
		string decimalToBinary(int n, bool flag = 0)
		{ // flag是否返回倒转 如4:100 4:001
			if (n == 0)
				return "0";
			string binaryNumber = "";
			while (n > 0)
			{
				binaryNumber = to_string(n % 2) + binaryNumber;
				n /= 2;
			}
			if (!flag)
				return binaryNumber;
			else
			{
				reverse(binaryNumber.begin(), binaryNumber.end());
				return binaryNumber;
			}
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

		EulerPrime(int n)
		{
			prime.assign(n + 1, 0);
			vis.assign(n + 1, 0);
			init(n);
		}

		bool isPrime(int n)
		{
			if (n == 1 || n == 0)
				return 0;
			return !vis[n];
		}

		map<int, int> primeFactorsMAP(int n)
		{ // 欧拉筛优化分解质因数 O(logn)
			map<int, int> mp;
			int m = n;
			for (int i : prime)
			{
				while (m % i == 0)
				{
					m /= i;
					mp[i]++;
				}
				if (isPrime(m) or m == 1) break;
			}
			if (m > 1)
				mp[m]++;
			return mp;
		}

		vector<int> primeFactorsVEC(int n)
		{ // 欧拉筛优化分解质因数 O(logn)
			vector<int> se;
			int m = n;
			for (int i : prime)
			{
				if (m % i == 0)se.push_back(i);
				while (m % i == 0)m /= i;
				if (isPrime(m) or m == 1) break;
			}
			if (m > 1)
				se.push_back(m);
			return se;
		}

	private:
		vector<bool> vis;
		void init(int n) // 欧拉筛
		{
			for (int i = 2; i <= n; i++)
			{
				if (!vis[i])
					prime[cnt++] = i;
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

		//int query_kth(int p, int pl, int pr, ll x) {
		//	/**
		//	* 返回最大的mid满足sum[1..mid] < x
		//	*/

		//	if (info[p].sum < x)return pr + 1;
		//	if (pl == pr)return pl;

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

	template <class Info, class Tag>
	class LazySegmentTree
	{
		/* 该模板可能在直接相加的情况下比较好用，区间替换可能会出大大小小问题，这里写出区间替换模板
		*  警惕当前 tag 为 Tag() 的情况，即 Tag() 为空不更新
		*  区间替换并求区间和板子
			struct Tag {
				ll newValue = 0;
				bool f = 0;  // 是否进行替换
				Tag() {};
				Tag(ll newValue, bool f = 0) :newValue(newValue), f(f) {};
				void apply(Tag t) {
					if (t.f) { // 这里很重要，因为这个板子初始化的时候就会调用 tag，所以这里上个 f 作为保险
						f = 1;
						newValue = t.newValue;
					}
				}
			};
			struct Info {
				ll value = 0;
				Info() {};
				Info(ll value) :value(value) {};
				void apply(Tag t, int len) {
					if (t.f)
						value = t.newValue * len;
				}
			};
			Info operator+(Info a, Info b) {
				Info c;
				c.value = a.value + b.value;
				return c;
			}

			void solve(int CASE)
			{
				int n; cin >> n;
				auto a = vector<Info>(n + 1, Info());
				fa(i, 1, n) {
					ll now; cin >> now;
					a[i].now = now;
				}

				// 初始化的时候自动会调用所有 Tag
				MT::LazySegmentTree<Info, Tag> sg(a);

				// 查看每个值
				fa(i, 1, n) {
					cout << sg.query(i, i).value << ' ';
				}
				cout << endl;

				int q; cin >> q;
				while (q--) {
					ll l, r, x; cin >> l >> r >> x;
					sg.modifyRange(l, r, Tag(x, 1));

					// 查看更新后的每个值
					fa(i, 1, n)
						cout << sg.query(i, i).value << ' ';
					cout << endl;
				}
				return;
			}
		*/
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
				info[p] = v;
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

	template <class T>
	class List
	{
	private:
		vector<int> l, r;
		map<T, int> pos;
		int cnt = 1, tail = 0;

	public:
		vector<T> value;

		List(int n = 2e5 + 10)
		{ // 预留空间，默认2e6
			value.assign(n * 10, 0);
			l.assign(n * 10, 0);
			r.assign(n * 10, 0);
		}

		void remove(int idx)
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

			// 插入节点
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

			// 插入节点
			l[cnt] = tail;
			r[cnt] = r[tail];
			l[r[tail]] = cnt;
			r[tail] = cnt++;

			tail++;
		}

		void print_all()
		{
			int k = tail;
			for (int i = r[0]; k; i = r[i])
			{
				cout << value[i] << ' ';
				k--;
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
		vector<int> lengths; // 以每个字符为中心的最长长度回文串的半径
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
		// vector<int>w; 树上边差分没准能用上，w[to] 表示 to 到其 fa 节点的边

		LCA(int n, vector<vector<int>>& e)
		{
			f.assign(n + 10, vector<int>(31));
			depth.assign(n + 10, 0);
			// w.assign(n + 10, 0);

			init(1, 0, e); // 邻接表
		}

		int lca(int x, int y)
		{
			if (depth[x] < depth[y])
				swap(x, y);
			// 后面进行x节点默认比y节点深
			for (int i = 29; i >= 0; i--)
				if (depth[x] - (1 << i) >= depth[y])
					x = f[x][i];
			if (x == y)
				return x; // 特判：y就是原本x的祖宗
			for (int i = 29; i >= 0; i--)
				if (f[x][i] != f[y][i]) // 说明还没找到祖宗，更新a、b后接着跳
					x = f[x][i], y = f[y][i];
			return f[x][0];
		}

		int dis(int x, int y) {
			int f = lca(x, y);
			return depth[x] - depth[f] + depth[y] - depth[f];
		}

	private:
		vector<vector<int>> f; // f[x][i]即x的第2^i个祖先 (31是倍增用的，最大跳跃为2^30)

		void init(int now, int fa, vector<vector<int>>& e) // 邻接表
		{
			depth[now] = depth[fa] + 1;
			f[now][0] = fa;                                 // 第一个祖先
			for (int i = 1; (1 << i) <= depth[now]; i++) // 求now的各个祖先
				f[now][i] = f[f[now][i - 1]][i - 1];
			for (int to : e[now])
			{ // now这个的点的祖先都找完了，dfs处理别的点
				if (to == fa)
					continue;
				init(to, now, e);
				// w[to] = e[i].w;
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
	private:
		vector<long long> a; // 线性基基底
		const int MN = 62;
		bool flag = false;

	public:
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
			for (int i = MN; ~i; i--)
				if (x & (1ll << i))
					if (!a[i]) { a[i] = x; return; }
					else x ^= a[i];
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
			* 双版本差分查询区间值域大于 k 的数的总数 sum
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
			* 双版本差分查询区间值域小于 k 的数的总数 sum
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
		/**
		* 有时会输出 -0，为正常现象，有必要可以特判掉。
		*/

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

			double len() const { return sqrt(len2()); } // 向量长度
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
			T Perimeter() {
				T ans = 0;
				int n = this->size();
				for (int i = 0; i < n; i++)
					ans += dist((*this)[i], (*this)[(i + 1) % n]);
				return ans;
			}

			// 多边形的面积
			T Area() {
				T area = 0;
				int n = this->size();
				for (int i = 0; i < n; i++) {
					area += cross((*this)[i], (*this)[(i + 1) % n]);
				}
				return abs(area) / 2.0;
			}

			// 多边形的面积 * 2
			ll Area2() {
				ll area = 0;
				int n = this->size();
				for (int i = 0; i < n; i++) {
					area += cross((*this)[i], (*this)[(i + 1) % n]);
				}
				return abs(area);
			}

			// atan2 极角排序，默认逆时针排序
			void Polar_angle_sort_atan2(const Point<T>& reference = Point<T>(0, 0)) {
				sort(this->begin(), this->end(),
					[&](const Point<T>& a, const Point<T>& b)->bool
					{ return a.polar_angle(reference) < b.polar_angle(reference); });
			}

			// cross 极角排序，默认逆时针排序
			void Polar_angle_sort_cross(const Point<T>& reference = Point<T>(0, 0)) {
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


		template<typename T>
		using Vector = Point<T>;

		template<typename T>
		using Segment = Line<T>;

		template<typename T>
		using PointSet = Polygon<T>;


		template<typename T>
		double dist(const Point<T>& A, const Point<T>& B) {
			/**
			* 两点距离
			*/
			return sqrt((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y));
		}

		template<typename T>
		ll dist2(const Point<T>& A, const Point<T>& B) {
			/**
			* 两点距离的平方
			*/
			return (A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y);
		}

		template<typename T>
		T dot(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 计算点积  a · b = |a| |b| cos
			* 可用于判断两向量夹角
			*/
			return A.x * B.x + A.y * B.y;
		}

		template<typename T>
		T cross(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 计算叉积  a · b = |a| |b| sin
			* 可以判断两向量的相对方向
			* 也能算两向量形成的平行四边形的有向面积
			*/
			return A.x * B.y - A.y * B.x;
		}

		template<typename T>
		double len(const Vector<T>& A) {
			/**
			* 向量长度
			*/
			return sqrt(dot(A, A));
		}

		template<typename T>
		T len2(const Vector<T>& A) {
			/**
			* 向量长度的平方
			*/
			return dot(A, A);
		}

		template<typename T>
		double angle(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 两向量夹角 (弧度制)
			*/
			return acos(dot(A, B) / len(A) / len(B));
		}

		Point<db> angle_to_point(const db& ang) {
			/**
			* 极角变单位坐标
			*/
			return { cos(ang), sin(ang) };
		}

		template<typename T>
		T area_parallelogram(const Point<T>& A, const Point<T>& B, const Point<T>& C) {
			/**
			* 计算两向量构成的平行四边形有向面积
			* 三个点A、B、C，以 A 为公共点，得到 2 个向量 AB 和 AC，它们构成的平行四边形
			* 请逆时针输入 B C 两点
			*/
			return cross(B - A, C - A);
		}

		template<typename T>
		T area_parallelogram(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 计算两向量构成的平行四边形有向面积
			* 两个有公共点的向量 A B 构成的平行四边形
			* 请逆时针输入 A B 向量
			*/
			return cross(A, B);
		}

		template<typename T>
		T area_triangle(const Point<T>& A, const Point<T>& B, const Point<T>& C) {
			/**
			* 计算两向量构成的三角形有向面积
			* 三个点A、B、C，以 A 为公共点，得到 2 个向量 AB 和 AC，它们构成的三角形
			* 请逆时针输入 B C 两点
			*/
			return cross(B - A, C - A) / 2.0;
		}


		template<typename T>
		T area_triangle(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 计算两向量构成的三角形有向面积
			* 两个有公共点的向量 A B 构成的三角形
			* 请逆时针输入 A B 向量
			*/
			return cross(A, B) / 2.0;
		}


		template<typename T>
		Vector<T> rotate(const Vector<T>& A, double rad) {
			/**
			* 向量旋转 (弧度制)
			* 特殊情况是旋转90度：
			* 逆时针旋转90度：Rotate(A, pi/2)，返回Vector(-A.y, A.x)；
			* 顺时针旋转90度：Rotate(A, -pi/2)，返回Vector(A.y, - A.x)。
			*/
			return Vector<T>(A.x * cos(rad) - A.y * sin(rad), A.x * sin(rad) + A.y * cos(rad));
		}


		template<typename T>
		Vector<T> normal(const Vector<T>& A) {
			/**
			* 单位法向量
			* 有时需要求单位法向量，即逆时针转90度，然后取单位值。
			*/
			return Vector<T>(-A.y / len(A), A.x / len(A));
		}

		template<typename T>
		bool vector_vector_parallel(const Vector<T>& A, const Vector<T>& B) {
			/**
			* 两个向量是否平行或重合
			*/
			return sgn(cross(A, B)) == 0;
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

			return fabs(cross(p - v.p1, v.p2 - v.p1)) / dist(v.p1, v.p2);
		}

		template<typename T>
		Point<T> point_line_proj(const Point<T>& p, const Line<T>& v) {
			/**
			* 点在直线上的投影点
			*/

			double k = dot(v.p2 - v.p1, p - v.p1) / len2(v.p2 - v.p1);
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
				return min(dist(p, v.p1), dist(p, v.p2));
			return point_line_dis(p, v);
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
				ans += dist(p[i], p[(i + 1) % n]);
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

			double dst = dist(p, C.c);
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
			Point<T> n = (v.p2 - v.p1) / len(v.p2 - v.p1);     // 单位向量
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
				db d1 = dist(p[i], p[j]);
				if (d1 > mx) {
					mx = d1;
					a = p[i];
					b = p[j];
				}
				db d2 = dist(p[(i + 1) % n], p[j]);
				if (d2 > mx) {
					mx = d2;
					a = p[(i + 1) % n];
					b = p[j];
				}
			}
			return mx;
		}


		template<typename T>
		ll farthest_point_to_point_dis2(const Polygon<T>& p, Point<T>& a, Point<T>& b) {
			/**
			* 旋转卡壳求最远点对及其距离
			* p 为凸包，ab 为最远点对
			* 返回值为最远点对的距离
			*/

			ll mx = 0;
			int n = p.size();
			// 经典单调性，双指针枚举即可
			int j = 1;
			for (int i = 0; i < n; i++) { // 实际在枚举边
				// 利用面积判断离当前边最远的点
				while (abs(area_parallelogram(p[(i + 1) % n], p[i], p[(j + 1) % n])) >
					abs(area_parallelogram(p[(i + 1) % n], p[i], p[j]))) {
					j = (j + 1) % n;
				}
				ll d1 = dist2(p[i], p[j]);
				if (d1 > mx) {
					mx = d1;
					a = p[i];
					b = p[j];
				}
				ll d2 = dist2(p[(i + 1) % n], p[j]);
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
				db d1 = dist(p[i], p[j]);
				db d2 = dist(p[(i + 1) % n], p[j]);
				mx = max({ d1, d2, mx });
			}
			return mx;
		}


		template<typename T>
		ll farthest_point_to_point_dis2(const Polygon<T>& p) {
			/**
			* 旋转卡壳求最远点对及其距离
			* p 为凸包，ab 为最远点对
			* 返回值为最远点对的距离
			*/

			ll mx = 0;
			int n = p.size();
			// 经典单调性，双指针枚举即可
			int j = 1;
			for (int i = 0; i < n; i++) { // 实际在枚举边
				// 利用面积判断离当前边最远的点
				while (abs(area_parallelogram(p[(i + 1) % n], p[i], p[(j + 1) % n])) >
					abs(area_parallelogram(p[(i + 1) % n], p[i], p[j]))) {
					j = (j + 1) % n;
				}
				ll d1 = dist2(p[i], p[j]);
				ll d2 = dist2(p[(i + 1) % n], p[j]);
				mx = max({ d1, d2, mx });
			}
			return mx;
		}
	}


	template <typename T, size_t N>
	class Trie01 {
	private:
		struct TrieNode {
			array<int, 2> children; // 指向孩子的 id
			/* 可再添些变量，比如 区间查询问题用到的 边归属：int id； */
		};

		// N：总共 N 个数
		unique_ptr<array<TrieNode, N>> t;

		int newNode() { return ++nodeCnt; }

		int MAX; 	      // MAX：数的最大二进制位数
		int root = 1;
		int nodeCnt = 1;
	public:
		Trie01(int MAX) :MAX(MAX), t(make_unique<array<TrieNode, N* MAX>>()) {}

		void insert(T num) {
			int now = root;
			// 枚举二进制数位
			for (int i = MAX; i >= 0; i--) {
				bool bit = (num >> i) & 1; // 当前数位
				if (not (*t)[now].children[bit]) {
					// 没有对应数位的边，则自己造边
					(*t)[now].children[bit] = newNode();
				}
				now = (*t)[now].children[bit];
			}
		}

		// 看看 num 跟 01 trie 里的哪个元素 XOR 最后的值最大
		T find_max_xor(T num) {
			int now = root;
			T maxXor = 0;
			for (int i = MAX; i >= 0; i--) {
				bool bit = (num >> i) & 1;
				// 尽量选与自己相反的，从而进 1
				if ((*t)[now].children[!bit]) {
					maxXor |= (1ll << i);
					now = (*t)[now].children[!bit];
				}
				else {
					// 没有的话只能乖乖往下走，这一位变 0
					now = (*t)[now].children[bit];
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
				if ((*t)[now].children[bit]) {
					now = (*t)[now].children[bit];
				}
				else {
					minXor |= (1ll << i);
					now = (*t)[now].children[bit];
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

}

namespace MT = MyTools;
using Math = MT::Math<ll>;
//using mint = MT::ModInt<998244353>;

/*
struct Tag {
	Tag() {}
	// is_init(0)

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
};
Info operator+(const Info& a, const Info& b) {
	Info c;
	return c;
}


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

#endif