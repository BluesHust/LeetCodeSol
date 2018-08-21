class BinaryFind:

    #将查找区间[first,last]划分为
    #[first,mid),[mid,mid],(mid,last]三个区间查找
    def solution(self,nums,target):
        '''
        如果查找成功，返回target的下标
        如果查找失败，返回-1
        :param nums:
        :param target:
        :return:
        '''
        #nums已有序
        first,last=0,len(nums)-1 #查找区间[first,last]
        while first<=last:
            mid=first+(last-first)//2
            print("first={} mid={} last={}".format(first,mid,last))
            if nums[mid]==target:
                return mid
            elif target<nums[mid]:
                last=mid-1
            else:
                first=mid+1
        return -1

    def recursiveSolution(self,nums,target):
        return self._recursiveSolution(nums,0,len(nums)-1,target)

    def _recursiveSolution(self,nums,first,last,target):
        '''
        在nums数组[first,last]区间内查找target
        :param nums:list
        :param first:
        :param last:
        :param target:
        :return:
        '''
        if first>last:return -1
        mid=first+(last-first)//2
        if target==nums[mid]:return mid
        elif target<nums[mid]:
            return self._recursiveSolution(nums,first,mid-1,target)
        else:
            return self._recursiveSolution(nums,mid+1,last,target)


    def run(self):
        nums=[10,11,12,16,18,23,29,33,48,54,57,68,77,84,98]
        print([i for i in range(len(nums))])
        print(nums)
        target=23
        print("{} at {}".format(target,self.solution(nums,target)))
        print("recursive {} at {}".format(target, self.recursiveSolution(nums, target)))
        target=50
        print("{} at {}".format(target, self.solution(nums, target)))
        print("recursive {} at {}".format(target, self.recursiveSolution(nums, target)))
        print(nums)

def  main():
    sol=BinaryFind()
    sol.run()

if __name__ == '__main__':
    main()