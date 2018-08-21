import operator
from collections import defaultdict


#2.1.11 删除元素
class RemoveElement:
    #删除数组中所有值为target的实例并返回数组新长度
    def solution(self,seq,target):
        index=0
        for i in range(len(seq)):
            if seq[i]!=target:
                seq[index]=seq[i]
                index+=1
        return index

    def run(self):
        seq1=[1,2,3,5,1,1,2,3,1]
        seq2=['a','b','a']
        print(seq1)
        newLen=self.solution(seq1,1)
        print(seq1[:newLen])
        print(seq2)
        newLen=self.solution(seq2,'a')
        print(seq2[:newLen])

#2.1.1 从有序数组中删除多余的元素（相同元素只保留一个）
class RemoveDuplicates01:

    def solution(self,nums:list):
        last=-1 #新数组的最后一个元素
        for i in range(len(nums)):
            if nums[i]!=nums[last]:
                last+=1
                nums[last]=nums[i]
        nums[:]=nums[:last+1]

    def run(self):
        a=[1,1,2,2,3,3,4,4]
        print(a)
        self.solution(a)
        print(a)

#2.1.2 移除重复的元素，但是相同元素允许出现两次
class RemoveDuplicates02:

    #nums已有序
    def orderedSolution(self, nums):
        if len(nums)<3:return len(nums)
        last=1 #新数组最后一个元素
        for i in range(2,len(nums)):
            if nums[i]!=nums[last-1]:
                last+=1
                nums[last]=nums[i]
        nums[:]=nums[:last+1]


    #nums无序
    def unorderedSolution(self,nums):
        #用字典来保存num在新数组中出现的次数
        num_times=defaultdict(int)
        next=0 #新数组的一下个元素索引
        for i in range(len(nums)):
            num_times[nums[i]]+=1
            if num_times[nums[i]]<=2:
                nums[next]=nums[i]
                next+=1
        nums[:]=nums[:next]

    def run(self):
        nums=[1,1,1,2,2,3,3,4,4,4]
        print(nums)
        self.orderedSolution(nums)
        print(nums)

        nums2=[1,2,2,2,1,1,3,3,1,4,3]
        print(nums2)
        self.unorderedSolution(nums2)
        print(nums2)


#2.1.3 Search in Rotated Sorted Array
class SearchInRotatedSortedArray:

    def solution(self,nums,target):
        #nums中不重复
        first,last=0,len(nums)-1 #查找区间[first,last]
        while first<=last:
            mid=first+(last-first)//2
            print("first={} mid={} last={}".format(first,mid,last))
            if nums[mid]==target:
                return mid
            # 只要target在数组中，其必在某个有序子区间内
            #[first,mid),(mid,last]中必有一个是有序的
            #只能在有序区间内进行二分查找，mid将查找区
            # 间分出一个有序区间，只需在该区间内查找target
            if nums[first]<nums[mid]: #[first,mid)为有序
                if nums[first]<=target and target<nums[mid]:
                    last=mid-1
                else:#target不在该有序区间
                    first=mid+1
            else: #(mid,last]为有序区间
                if nums[mid]<target and target<=nums[last]:
                    first=mid+1
                else:
                    last=mid-1
        return -1

    def run(self):
        nums=[68,77,84,98,10,11,12,16,18,23,29]
        print(nums)
        target = 23
        print("{} at {}".format(target, self.solution(nums, target)))
        target = 50
        print("{} at {}".format(target, self.solution(nums, target)))

        nums = [1,1,1,1,1,3,1,1]
        print(nums)
        target = 3
        print("{} at {}".format(target, self.solution(nums, target)))













#2.1.7  Two sum两数之和
#给定一个整数数组和一个目标值，找出数组中和为目标值的两个数。
#你可以假设每个输入只对应一种答案，且同样的元素不能被重复利用。
class TwoSumSolution:
    # brute暴力求解O(n^2)
    def brute(self,nums,target):
        for i in range(len(nums)):
            for j in range(i,len(nums)):
                if nums[i]+nums[j]==target:
                    return i,j

    #使用字典，建立数字到映射坐标的映射,遍历数组O(n)
    #如果target-nums[i]在字典中，则完成查找
    #如果target-nums[i]不在字典中，则将nums[i]:i加入字典
    def useHash(self,nums,target):
        num_id_dict={}
        for i in range(len(nums)):
            if target-nums[i] in num_id_dict:
                return num_id_dict[target-nums[i]],i
            else:
                num_id_dict[nums[i]]=i

    #对数组进行排序，用两个指针head、tail进行夹逼
    #if sum=target:完成查找
    #if sum>target:tail-=1
    #if sum<target:head+=1
    def sandwich(self,nums,target):
        id_num=list(enumerate(nums))
        print(id_num)
        #id_num.sort(key=lambda x:x[1]) #根据num进行排序
        id_num.sort(key=operator.itemgetter(1))  #和上面一样，但是更快
        print(id_num)
        head,tail=0,len(id_num)-1
        while head<tail:
            sum=id_num[head][1]+id_num[tail][1]
            if sum==target:
                return id_num[head][0],id_num[tail][0]
            elif sum>target:
                tail-=1
                #跳过重复的数字
                while id_num[tail][1]==id_num[tail+1][1] and head<tail:
                    continue
            else:
                head+=1
                #跳过重复的数字
                while id_num[head][1]==id_num[head-1][1] and head<tail:
                    continue

    def run(self):
        nums=[1,4,6,3,6,9]
        target=9
        print("brute:{},{}".format(*self.brute(nums,target)))
        print("useHeah:{},{}".format(*self.useHash(nums, target)))
        print("sandwich:{},{}".format(*self.sandwich(nums, target)))

#2.1.8 在数组中找出三个数之和等于target,找出所有的组合，但不能重复
#先排序，做一次循环，每次循环先选定一个数，再利用夹逼
class ThreeSumSolution:
    #一层循环，在循环内部进行左右夹逼
    def three_sum(self,nums,target):
        nums=sorted(nums)
        for i in range(len(nums)-2):
            #选定其中一个数字为nums[i]，在(i,len(nums))区间进行夹逼
            if i>0 and nums[i]==nums[i-1]:continue
            print("num1={}".format(nums[i]))
            head,tail=i+1,len(nums)-1
            while head<tail:
                sum=nums[i]+nums[head]+nums[tail]
                if sum>target:
                    tail-=1
                    #可以添加一句跳过相同的数字
                    while head<tail and nums[tail]==nums[tail+1]:
                        tail-=1
                elif sum<target:
                    head+=1
                    while head<tail and nums[head]==nums[head-1]:
                        head+=1
                else:
                    return nums[i],nums[head],nums[tail]

    def run(self):
        nums=[9,4,6,87,2,4,7,4,1]
        target=19
        res=self.three_sum(nums,target)
        print("{}+{}+{}={}".format(*res,target))




#2.1.10 4Sum
class FourSolution:

    #时间复杂度O(n^3)，可能会超时
    def sandwichSolution(self,nums,target):
        nums=sorted(nums)
        res=[]
        for i in range(len(nums)-3):
            #跳过重复数字
            if i>0 and nums[i]==nums[i-1]:continue
            for j in range(i+1,len(nums)-2):
                #跳过重复数字
                if nums[j]==nums[j-1]:continue
                #做夹逼
                head,tail=j+1,len(nums)-1
                while head<tail:
                    sum=nums[i]+nums[j]+nums[head]+nums[tail]
                    if sum>target:
                        tail-=1
                        while head<tail and nums[tail]==nums[tail+1]:
                            tail-=1
                    elif sum<target:
                        head+=1
                        while head<tail and nums[head]==nums[head-1]:
                            head+=1
                    else:
                        res.append((nums[i],nums[j],nums[head],nums[tail]))
                        head+=1
                        tail-=1
        return res


    def mapSolution(self,nums,target):
        nums=sorted(nums)
        #先储存任意两个数的和
        cache=defaultdict(list)
        res=[]
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                cache[nums[i]+nums[j]].append((i,j))

        for c in range(len(nums)):
            for d in range(c+1,len(nums)):
                if target-nums[c]-nums[d] not in cache:
                    continue
                for a_b in cache[target-nums[c]-nums[d]]:
                    #b<c
                    if c <= a_b[1]:continue
                    res.append((nums[a_b[0]],nums[a_b[1]]
                                ,nums[c],nums[d]))

        #去除重复项
        return sorted(set(res),key=res.index)

    def run(self):
        nums=[1,0,-1,0,-2,-2,2]  #两个-2会导致重复
        target=0
        res=self.mapSolution(nums,target)
        print(res)

#2.1.12
class NextPermutation:

    def solution(self,nums):
        first_vio=-1 #第一个破坏从右往左升序规则的坐标
        #找到first_vio
        for i in range(len(nums)-1,0,-1):
            if nums[i-1]<nums[i]:
                first_vio=i-1
                break
        if first_vio==-1: #如果已经全部升序
            nums.reverseBetween()
            return None
        #找到第一个大于nums[first_vio]的数并交换
        for i in range(len(nums)-1,first_vio,-1):
            if nums[i]>nums[first_vio]:
                nums[first_vio],nums[i]=nums[i],nums[first_vio]
                break
        #反转对first_vio右边的序列
        nums[first_vio+1:]=nums[len(nums)-1:first_vio:-1]
        return None

    def run(self):
        sample=[[1,2,3],[3,2,1],[1,1,5]]
        for nums in sample:
            print(nums)
            self.solution(nums)
            print(nums)

# 2.1.15 接雨水
class TrapWater:

    #思路1:找出每个柱子两端的最高柱子
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left_cache = []
        curMaxLeft = 0
        curMaxRight = 0
        right_cache = []
        for i in range(len(height)):
            left_cache.append(curMaxLeft)
            right_cache.append(curMaxRight)
            if height[i] > curMaxLeft:
                curMaxLeft = height[i]
            if height[len(height)-1-i]>curMaxRight:
                curMaxRight=height[len(height)-1-i]
        right_cache[:]=right_cache[::-1]
        water = 0
        for i in range(1, len(height) - 1):  # 首尾柱子两个不能装水
            if min(left_cache[i], right_cache[i]) > height[i]:
                water += (min(left_cache[i], right_cache[i]) - height[i])
        return water

    #思路2;先找到最高的柱子，以此为界限，将数组分成前后两半处理
    def trap2(self,height):
        size=len(height)
        max=0
        for i in range(size):
            if height[i]>height[max]:
                max=i
        peak=0
        sum=0
        for i in range(0,max):
            if height[i]>peak:
                peak=height[i]
            else:
                sum+=peak-height[i]
        peak=0
        for i in range(size-1,max,-1):
            if height[i]>peak:
                peak=height[i]
            else:
                sum+=peak-height[i]
        return sum






    def run(self):
        height=[0,1,0,2,1,0,1,3,2,1,2,1]
        water=self.trap(height)
        water2=self.trap2(height)
        print("water={}".format(water))
        print("water={}".format(water2))


# 2.1.16顺时针旋转图像90度
class RotateImage:

    #思路1：先沿中线翻转，再沿主对角线翻转
    def rotate(self,matrix):
        '''

        :type matrix:llst[list[int]]
        :return:
        '''
        theRow=len(matrix)
        theCol=len(matrix[0])
        mid=len(matrix)//2
        for row in range(mid): #沿中线反转
            matrix[row],matrix[theRow-1-row]=matrix[theRow-1-row],matrix[row]
        for row in range(theRow-1):
            for col in range(row,theCol):
                matrix[row][col],matrix[col,row]=matrix[col,row],matrix[row][col]



# 2.1.17 加一
class PlusOne:

    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry = 1  # 进位
        for i in range(len(digits) - 1, -1, -1):
            carry,mod=divmod(digits[i]+carry,10)
            digits[i]=mod
        if carry > 0:  # 最高位产生了进位
            digits.insert(0,carry)

    def run(self):
        nums=[9,9,9]
        self.plusOne(nums)
        print(nums)

# 2.1.18爬楼梯
class ClimbingStairs:

    def dp_climbing(self,n):
        cache=[1 for i in range(n+1)]
        for i in range(2,n+1):
            cache[i]=cache[i-1]+cache[i-2]
        return cache[n]

    def loop_climbing(self,n):
        prev=1 #n=0时只有一种方法
        cur=1 #n=1时有一种方法
        tmp=0
        for i in range(2,n):
            tmp=cur
            cur+=prev
            prev=tmp
        return cur


class GasStation:

    def bruteSolution(self,gas,cost):
        n=len(gas)
        for i in range(n): #模拟n个站
            sum=0
            for j in range(n): #模拟从第i个站出发
                sum+=gas[(i+j)%n]-cost[(i+j)%n]
                if sum<0:break
            if sum>=0:return i #找到起点
        return -1 #没找到起点

    def solution(self,gas,cost):
        n=len(gas)
        j=0
        sum=0 #[j,i]区间的净输入
        total=0 #整个区间的净输入
        for i in range(n):
            sum+=gas[i]-cost[i]
            total+=gas[i]-cost[i]
            if sum<0: #[j,i]区间的净输入为负，在此区间发车不能成功
                j=i+1
                sum=0
        return j if total>=0 else -1


# 2.1.22 candy给小朋友发糖果
class Candy:

    def solution(self,ratings):
        n=len(ratings)
        increments=[0 for i in range(n)]
        for i in range(1,n): #从左到右遍历一遍
            if ratings[i]>ratings[i-1]: #如果比左边的大，应该多发一颗
                increments[i]=increments[i-1]+1
        inc=1
        for i in range(n-2,-1,-1):
            if ratings[i]>ratings[i+1]:
                increments[i]=max(inc,increments[i])
                inc+=1
            else:
                inc=1
        return sum(increments)+n

    def run(self):
        rates=[1,0,2]
        print(self.solution(rates))







def main():
    sol=Candy()
    sol.run()

if __name__ == '__main__':
    main()




